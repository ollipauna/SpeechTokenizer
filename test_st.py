import torch
import os
from speechtokenizer import SpeechTokenizer


def test_st():
    # Instantiate a model
    path = os.environ.get('SPEECH_TOKENIZER_PATH', None)

    if path is None:
        raise RuntimeError("Environment variable SPEECH_TOKENIZER_PATH is not set! "
                            "Please set it to the path where the model config should be stored ")


    config_path = 'config.json'
    ckpt_path = 'SpeechTokenizer.pt'

    config_path = os.path.join(path, config_path) 
    ckpt_path = os.path.join(path, ckpt_path) 

    model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)

    # Create a random audio signal
    x = torch.randn(2, 1, 16000, requires_grad=True)
    model.eval()
    x_org_quantized = model.decode(model.encode(x))
    model.train()
    x_org_quantized = model.decode(model.encode(x))

    for _ in range(10):
    
        x_quantized, _, _ = model(x)

        # Check that the audio reconstructions match
        assert torch.allclose(x_quantized, x_org_quantized, atol=1e-3)
        assert x_quantized.size() == x.size()

    loss = torch.pow(x_quantized, 2).sum()
    loss.backward()
    model.zero_grad()

    # Check that gradients are being tracked
    assert x.grad is not None

    # Check that gradient values look sensible
    print(x.grad)

    # Check that adding noise changes the gradients
    x_other = x.detach() + (0.1**0.5) * torch.randn(1, 1, 16000)
    x_other.requires_grad_()
    x_quantized_other , _, _ = model(x_other)
    loss_other = torch.pow(x_quantized_other, 2).sum()
    loss_other.backward()
    print(torch.mean(torch.abs(x.grad - x_other.grad)))


if __name__ == "__main__":
    test_st()
