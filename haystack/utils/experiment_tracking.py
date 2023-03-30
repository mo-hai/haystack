from typing import Optional, Any, Dict, Union

from abc import ABC, abstractmethod
import logging
from pathlib import Path

import mlflow
from requests.exceptions import ConnectionError

from clearml import Task, Logger

from haystack import __version__
from haystack.environment import get_or_create_env_meta_data


logger = logging.getLogger(__name__)


def flatten_dict(dict_to_flatten: dict, prefix: str = ""):
    flat_dict = {}
    for k, v in dict_to_flatten.items():
        if isinstance(v, dict):
            flat_dict.update(flatten_dict(v, prefix + k + "_"))
        else:
            flat_dict[prefix + k] = v
    return flat_dict


class BaseTrackingHead(ABC):
    """
    Base class for tracking experiments.

    This class can be extended to implement custom logging backends like MLflow, WandB, or TensorBoard.
    """

    @abstractmethod
    def init_experiment(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        nested: bool = False,
    ):
        raise NotImplementedError()

    @abstractmethod
    def track_metrics(self, metrics: Dict[str, Any], step: int):
        raise NotImplementedError()

    @abstractmethod
    def track_artifacts(self, dir_path: Union[str, Path], artifact_path: Optional[str] = None):
        raise NotImplementedError()

    @abstractmethod
    def track_params(self, params: Dict[str, Any]):
        raise NotImplementedError()

    @abstractmethod
    def end_run(self):
        raise NotImplementedError()


class NoTrackingHead(BaseTrackingHead):
    """
    Null object implementation of a tracking head: i.e. does nothing.
    """

    def init_experiment(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        nested: bool = False,
    ):
        pass

    def track_metrics(self, metrics: Dict[str, Any], step: int):
        pass

    def track_artifacts(self, dir_path: Union[str, Path], artifact_path: Optional[str] = None):
        pass

    def track_params(self, params: Dict[str, Any]):
        pass

    def end_run(self):
        pass


class Tracker:
    """
    Facade for tracking experiments.
    """

    tracker: BaseTrackingHead = NoTrackingHead()

    @classmethod
    def init_experiment(
        cls,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        nested: bool = False,
    ):
        cls.tracker.init_experiment(experiment_name=experiment_name, run_name=run_name, tags=tags, nested=nested)

    @classmethod
    def track_metrics(cls, metrics: Dict[str, Any], step: int):
        cls.tracker.track_metrics(metrics=metrics, step=step)

    @classmethod
    def track_artifacts(cls, dir_path: Union[str, Path], artifact_path: Optional[str] = None):
        cls.tracker.track_artifacts(dir_path=dir_path, artifact_path=artifact_path)

    @classmethod
    def track_params(cls, params: Dict[str, Any]):
        cls.tracker.track_params(params=params)

    @classmethod
    def end_run(cls):
        cls.tracker.end_run()

    @classmethod
    def set_tracking_head(cls, tracker: BaseTrackingHead):
        cls.tracker = tracker


class StdoutTrackingHead(BaseTrackingHead):
    """
    Experiment tracking head printing metrics and params to stdout.
    Useful for services like AWS SageMaker, where you parse metrics from the actual logs
    """

    def init_experiment(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        nested: bool = False,
    ):
        logger.info("\n **** Starting experiment '%s' (Run: %s)  ****", experiment_name, run_name)

    def track_metrics(self, metrics: Dict[str, Any], step: int):
        logger.info("Logged metrics at step %s: \n %s", step, metrics)

    def track_params(self, params: Dict[str, Any]):
        logger.info("Logged parameters: \n %s", params)

    def track_artifacts(self, dir_path: Union[str, Path], artifact_path: Optional[str] = None):
        logger.warning("Cannot log artifacts with StdoutLogger: \n %s", dir_path)

    def end_run(self):
        logger.info("**** End of Experiment **** ")


class MLflowTrackingHead(BaseTrackingHead):
    def __init__(self, tracking_uri: str, auto_track_environment: bool = True) -> None:
        """
        Experiment tracking head for MLflow.
        """
        super().__init__()
        self.tracking_uri = tracking_uri
        self.auto_track_environment = auto_track_environment

    def init_experiment(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        nested: bool = False,
    ):
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=run_name, nested=nested, tags=tags)
            logger.info(
                "Tracking run %s of experiment %s by mlflow under %s", run_name, experiment_name, self.tracking_uri
            )
            if self.auto_track_environment:
                mlflow.log_params(flatten_dict({"environment": get_or_create_env_meta_data()}))
        except ConnectionError:
            raise Exception(
                f"MLflow cannot connect to the remote server at {self.tracking_uri}.\n"
                f"MLflow also supports logging runs locally to files. Set the MLflowTrackingHead "
                f"tracking_uri to an empty string to use that."
            )

    def track_metrics(self, metrics: Dict[str, Any], step: int):
        try:
            metrics = flatten_dict(metrics)
            mlflow.log_metrics(metrics, step=step)
        except ConnectionError:
            logger.warning("ConnectionError in logging metrics to MLflow.")
        except Exception as e:
            logger.warning("Failed to log metrics: %s", e)

    def track_params(self, params: Dict[str, Any]):
        try:
            params = flatten_dict(params)
            mlflow.log_params(params)
        except ConnectionError:
            logger.warning("ConnectionError in logging params to MLflow")
        except Exception as e:
            logger.warning("Failed to log params: %s", e)

    def track_artifacts(self, dir_path: Union[str, Path], artifact_path: Optional[str] = None):
        try:
            mlflow.log_artifacts(dir_path, artifact_path)
        except ConnectionError:
            logger.warning("ConnectionError in logging artifacts to MLflow")
        except Exception as e:
            logger.warning("Failed to log artifacts: %s", e)

    def end_run(self):
        mlflow.end_run()



class ClearMLTrackingHead(BaseTrackingHead):
    def __init__(self, tracking_uri: str = None, auto_track_environment: bool = True) -> None:
        """
        Experiment tracking head for MLflow.
        """
        super().__init__()
        self.tracking_uri = tracking_uri
        self.auto_track_environment = auto_track_environment
        self.task = None

    def init_experiment(
        self,
        experiment_name: str,  # project name of at least 3 characters
        run_name: str,  # task name of at least 3 characters
        tags: Optional[list[str]] = None,
    ):
        try:
            self.task = Task.init(
                project_name=experiment_name,
                task_name=run_name,
                tags=tags,
                output_uri=self.tracking_uri,
            )
            logger.info(
                "Tracking task %s of project %s by clearml. Artifacts logged to %s", run_name, experiment_name, self.tracking_uri
            )
            if self.auto_track_environment:
                self.task.set_user_properties(flatten_dict({"environment": get_or_create_env_meta_data()}))
        except ConnectionError:
            raise Exception(
                f"ClearML cannot connect to the remote server, something is wrong."
            )
        except Exception as e:
            logger.warning("Something failed: %s", e)

    def track_metrics(self, metrics: Dict[str, Any], step: int):
        try:
            metrics = flatten_dict(metrics)
            for k, v in metrics.items():
                Logger.current_logger().report_scalar(
                    title=k, series=k, iteration=step, value=v)
        except ConnectionError:
            logger.warning("ConnectionError in logging metrics to ClearML")
        except Exception as e:
            logger.warning("Failed to log metrics: %s", e)

    def track_params(self, params: Dict[str, Any]):
        try:
            params = flatten_dict(params)
            self.task.connect(params)
        except ConnectionError:
            logger.warning("ConnectionError in logging params to ClearML")
        except Exception as e:
            logger.warning("Failed to log params: %s", e)

    def track_artifacts(self, dir_path: Union[str, Path], artifact_path: Optional[str] = None):
        pass

    def end_run(self):
        self.task.close()
