import os
import sys
import unittest
from model_infer import test_model_inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestLlama2Infer(unittest.TestCase):

    def test_llama2_infer(self):
        from dancingmodel.models.llama2.model import Llama2TpPartModel
        test_model_inference(world_size=8, 
                             model_dir="/path/llama2-7b-chat", 
                             model_class=Llama2TpPartModel, 
                             batch_size=20, 
                             input_len=1024, 
                             output_len=1024)
        return


if __name__ == '__main__':
    unittest.main()