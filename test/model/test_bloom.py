import os
import sys
import unittest
from model_infer import test_model_inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestBloomInfer(unittest.TestCase):

    def test_bloom_infer(self):
        from dancingmodel.models.bloom.model import BloomTpPartModel
        test_model_inference(world_size=8, 
                             model_dir="/path/bloom-7b", 
                             model_class=BloomTpPartModel, 
                             batch_size=20, 
                             input_len=1024, 
                             output_len=1024)
        return


if __name__ == '__main__':
    unittest.main()
