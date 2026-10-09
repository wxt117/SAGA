from __future__ import annotations

import unittest

from saga.agent.intent_recognizer import recognize_request


class IntentRecognizerTest(unittest.TestCase):
    def test_english_traditional_sar_augmentation(self) -> None:
        result = recognize_request(
            "use traditional SAR augmentation on examples/demo_dataset and generate 12 samples",
            dataset_root="examples/demo_dataset",
        )

        intent = result["intent_spec"]["intent"]
        self.assertEqual(intent["task"], "traditional_augmentation")
        self.assertEqual(intent["target_count"], 12)

    def test_english_traditional_sar_data_augmentation(self) -> None:
        result = recognize_request(
            "apply traditional SAR data augmentation and create 8 images",
            dataset_root="examples/demo_dataset",
        )

        intent = result["intent_spec"]["intent"]
        self.assertEqual(intent["task"], "traditional_augmentation")
        self.assertEqual(intent["target_count"], 8)


if __name__ == "__main__":
    unittest.main()
