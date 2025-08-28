import unittest
import numpy as np
import bp_scratch as bp


def set_seed(seed=1337):
    np.random.seed(seed)


class TestBackpropScratch(unittest.TestCase):
    def test_linear_forward_shape(self):
        set_seed()
        lin = bp.Linear(5, 3)
        x = np.random.randn(7, 5)
        y = lin.forward(x)
        self.assertEqual(y.shape, (7, 3))

    def test_relu_backward_mask(self):
        relu = bp.ReLU()
        x = np.array([[-1.0, 0.0, 2.0]])
        y = relu.forward(x)
        self.assertTrue(np.allclose(y, [[0.0, 0.0, 2.0]]))
        grad_out = np.ones_like(x)
        dx = relu.backward(grad_out)
        self.assertTrue(np.allclose(dx, [[0.0, 0.0, 1.0]]))

    def test_linear_gradients_numeric(self):
        set_seed()
        lin = bp.Linear(4, 2, name="lin")
        x = np.random.randn(6, 4)
        y_true = np.random.randn(6, 2)
        loss = bp.MSELoss()

        # forward / backward (analytic)
        y_pred = lin.forward(x)
        val = loss.forward(y_pred, y_true)
        grad_out = loss.backward()
        lin.zero_grad()
        dx = lin.backward(grad_out)

        # finite-diff
        grads_fd, base_loss = bp.finite_diff_grad(lin, loss, x, y_true, eps=1e-6)

        self.assertAlmostEqual(val, base_loss, places=8)
        self.assertTrue(np.allclose(lin.W.grad, grads_fd["lin.weight"], atol=1e-5))
        if lin.b is not None:
            self.assertTrue(np.allclose(lin.b.grad, grads_fd["lin.bias"], atol=1e-5))

    def test_mlp_training_decreases_loss(self):
        set_seed()
        # tiny synthetic regression
        N, Din, H, Dout = 64, 5, 8, 3
        X = np.random.randn(N, Din)
        true_W1 = np.random.randn(H, Din)
        true_b1 = np.random.randn(H)
        true_W2 = np.random.randn(Dout, H)
        true_b2 = np.random.randn(Dout)

        # generate targets from a ground-truth MLP with ReLU
        Y = np.maximum(X @ true_W1.T + true_b1, 0.0) @ true_W2.T + true_b2

        model = bp.Sequential(
            bp.Linear(Din, H, name="l1"),
            bp.ReLU(),
            bp.Linear(H, Dout, name="l2")
        )
        loss = bp.MSELoss()
        opt = bp.SGD(model.parameters(), lr=1e-1, momentum=0.0)

        # measure initial loss
        y_pred0 = model.forward(X)
        val0 = loss.forward(y_pred0, Y)

        prev = None
        for t in range(2000):
            y_pred = model.forward(X)
            val = loss.forward(y_pred, Y)
            model.zero_grad()
            grad_out = loss.backward()
            model.backward(grad_out)
            opt.step()
            prev = val

        # success if we fit very well OR we reduce loss by 90%+
        self.assertTrue(val < 1e-2 or val < 0.1 * val0)

    def test_compare_with_pytorch_if_available(self):
        try:
            import torch
        except Exception:
            self.skipTest("PyTorch not available; skipping comparison test.")
            return

        import torch.nn as nn
        torch.set_default_dtype(torch.float64)
        rng = np.random.default_rng(42)

        Din, H, Dout, N = 4, 6, 3, 5
        W1 = rng.standard_normal((H, Din))
        b1 = rng.standard_normal((H,))
        W2 = rng.standard_normal((Dout, H))
        b2 = rng.standard_normal((Dout,))

        model = bp.Sequential(bp.Linear(Din, H, name="l1"), bp.ReLU(), bp.Linear(H, Dout, name="l2"))
        model.layers[0].W.data[...] = W1; model.layers[0].b.data[...] = b1
        model.layers[2].W.data[...] = W2; model.layers[2].b.data[...] = b2

        torch_model = nn.Sequential(nn.Linear(Din, H), nn.ReLU(), nn.Linear(H, Dout))
        with torch.no_grad():
            torch_model[0].weight.copy_(torch.tensor(W1)); torch_model[0].bias.copy_(torch.tensor(b1))
            torch_model[2].weight.copy_(torch.tensor(W2)); torch_model[2].bias.copy_(torch.tensor(b2))

        X = rng.standard_normal((N, Din))
        T = rng.standard_normal((N, Dout))

        loss_np = bp.MSELoss()
        Yp = model.forward(X)
        L_np = loss_np.forward(Yp, T)
        model.zero_grad()
        grad_out = loss_np.backward()
        dX_np = model.backward(grad_out)

        X_t = torch.tensor(X, requires_grad=True)
        T_t = torch.tensor(T)
        crit = nn.MSELoss(reduction='mean')
        Y_t = torch_model(X_t)
        L_t = crit(Y_t, T_t)
        L_t.backward()

        assert np.allclose(Yp, Y_t.detach().numpy(), atol=1e-8)
        assert abs(L_np - L_t.item()) < 1e-10
        assert np.allclose(dX_np, X_t.grad.detach().numpy(), atol=1e-6)
        assert np.allclose(model.layers[0].W.grad, torch_model[0].weight.grad.detach().numpy(), atol=1e-6)
        assert np.allclose(model.layers[0].b.grad, torch_model[0].bias.grad.detach().numpy(), atol=1e-6)
        assert np.allclose(model.layers[2].W.grad, torch_model[2].weight.grad.detach().numpy(), atol=1e-6)
        assert np.allclose(model.layers[2].b.grad, torch_model[2].bias.grad.detach().numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
