import os
import sys
import unittest
from model_infer import test_model_inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestInternlmInfer(unittest.TestCase):

    def test_internlm_infer(self):
        from dancingmodel.models.internlm.model import InternlmTpPartModel
        test_model_inference(world_size=8, 
                             model_dir="/path/internlm-chat-7b/", 
                             model_class=InternlmTpPartModel, 
                             batch_size=20, 
                             input_len=1024, 
                             output_len=1024)
        return


if __name__ == '__main__':
    unittest.main()