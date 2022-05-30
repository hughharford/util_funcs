import os

def test_no_helpful_bits():
    assert len(os.listdir('../util_funcs/.')) > 1
