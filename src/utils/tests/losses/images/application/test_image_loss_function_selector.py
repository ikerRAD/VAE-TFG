from typing import List, Any, Dict, Callable, Tuple
from unittest import TestCase
import tensorflow as tf
from unittest.mock import patch, Mock, call

from src.utils.losses.images.application.image_loss_function_selector import (
    ImageLossFunctionSelector,
    dkl_mse,
)


def mock_loss_function(
    z: tf.Tensor,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, Any]]:
    return 0.0, {"function_name": "mock_loss_function"}


class TestImageLossFunctionSelector(TestCase):
    def setUp(self) -> None:
        self.tensor_mock = Mock(spec=tf.Tensor)

    def test_possible_keys(self) -> None:
        expected_keys: List[str] = ["dkl_mse"]

        retrieved_keys: List[str] = ImageLossFunctionSelector.possible_keys()

        self.assertListEqual(expected_keys, retrieved_keys)

    @patch(
        "src.utils.losses.images.application.image_loss_function_selector.dkl_mse",
        return_value=(0.1, {"function_name": "dkl_mse"}),
    )
    def test_select(self, *_) -> None:
        scenarios: List[Dict[str, Any]] = [
            {
                "msg": "DKL with MSE loss function",
                "loss_function": "dkl_mse",
                "function_return_value": (0.1, {"function_name": "dkl_mse"}),
            },
            {
                "msg": "Custom loss function",
                "loss_function": mock_loss_function,
                "function_return_value": (0.0, {"function_name": "mock_loss_function"}),
            },
        ]
        for scenario in scenarios:
            with self.subTest(scenario["msg"]):
                retrieved_loss_function: Callable = ImageLossFunctionSelector.select(
                    scenario["loss_function"]
                )
                loss_function_value: Tuple[
                    float, Dict[str, Any]
                ] = retrieved_loss_function(
                    self.tensor_mock,
                    self.tensor_mock,
                    self.tensor_mock,
                    self.tensor_mock,
                    self.tensor_mock,
                )

                self.assertEqual(scenario["function_return_value"], loss_function_value)

    @patch(
        "src.utils.losses.images.application.image_loss_function_selector.tf.reduce_mean",
        return_value=0.0,
    )
    @patch(
        "src.utils.losses.images.application.image_loss_function_selector.tf.keras.losses.mean_squared_error",
        return_value=1.0,
    )
    @patch(
        "src.utils.losses.images.application.image_loss_function_selector.tf.reduce_sum",
        return_value=2.0,
    )
    @patch(
        "src.utils.losses.images.application.image_loss_function_selector.__log_normal_pdf"
    )
    def test_dkl_mse(
        self,
        log_normal_pdf_mock: Mock,
        reduce_sum_mock: Mock,
        mse_mock: Mock,
        *_,
    ) -> None:
        retrieved_value: Tuple[float, Dict[str, float]] = dkl_mse(
            self.tensor_mock,
            self.tensor_mock,
            self.tensor_mock,
            self.tensor_mock,
            self.tensor_mock,
        )

        self.assertEqual(
            retrieved_value, (0.0, {"logpz": 0.0, "logqz_x": 0.0, "mse": 0.0})
        )
        log_normal_pdf_mock.assert_has_calls(
            [
                call(self.tensor_mock, 0.0, 0.0),
                call(self.tensor_mock, self.tensor_mock, self.tensor_mock),
            ]
        )
        reduce_sum_mock.assert_called_with(1.0, axis=[1, 2])



        mse_mock.assert_called_with(self.tensor_mock, self.tensor_mock)
