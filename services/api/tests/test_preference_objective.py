import copy
import math

import pytest
torch = pytest.importorskip("torch")
from ai_home.foundation.model import FoundationConfig, FoundationModel
from ai_home.foundation.preference_objective import dpo_loss, response_log_probability


def test_dpo_reference_is_detached_and_has_correct_gradient():
    chosen = torch.tensor([-2.0, -3.0], requires_grad=True)
    rejected = torch.tensor([-3.0, -4.0], requires_grad=True)
    reference = torch.tensor([-2.0, -3.0], requires_grad=True)
    loss = dpo_loss(chosen, rejected, reference, reference - 1)
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert torch.all(chosen.grad < 0) and torch.all(rejected.grad > 0)
    assert reference.grad is None
    with pytest.raises(ValueError):
        dpo_loss(chosen, rejected[:1], reference, reference)
    with pytest.raises(ValueError):
        dpo_loss(chosen, rejected, reference, reference, beta=False)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_own_model_preference_update_and_response_mask(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = FoundationModel(FoundationConfig(vocab_size=262, width=16, layers=1,
            heads=2, kv_heads=1, hidden=32, context=24)).to(device)
    reference = copy.deepcopy(model).eval()
    before = {name: value.detach().clone() for name, value in reference.state_dict().items()}
    prompt, chosen, rejected = [257, 260, 65, 258, 261], [66, 258], [67, 258]
    with torch.no_grad():
        ref_c = response_log_probability(reference, prompt, chosen)
        ref_r = response_log_probability(reference, prompt, rejected)
        sequence = torch.tensor([prompt + chosen], device=device)
        logits, _ = model(sequence[:, :-1])
        manual = sum(logits[0, len(prompt)-1+i].log_softmax(-1)[token] for i, token in enumerate(chosen))
        assert ref_c.item() == pytest.approx(manual.item(), abs=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    initial = None
    for _ in range(12):
        optimizer.zero_grad()
        c, r = response_log_probability(model, prompt, chosen), response_log_probability(model, prompt, rejected)
        loss = dpo_loss(c, r, ref_c, ref_r)
        if initial is None:
            initial = (c-r).item()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = (response_log_probability(model, prompt, chosen)-response_log_probability(model, prompt, rejected)).item()
    assert final > initial + .1
    assert all(torch.equal(value, before[name]) for name, value in reference.state_dict().items())
    assert all(parameter.grad is None for parameter in reference.parameters())
    assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items())
    with pytest.raises(ValueError, match="context"):
        response_log_probability(model, prompt, chosen * 20)
