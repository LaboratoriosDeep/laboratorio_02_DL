"""Interfaces para la parte de incertidumbre del laboratorio."""

import torch
import torch.nn as nn


def enable_dropout_during_inference(model: nn.Module) -> None:
    """
    Activa las capas Dropout durante inferencia.

    model.eval() desactiva el dropout por defecto. Este paso lo reactiva
    explicitamente para que cada pasada forward sea estocastica.
    """

    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module, inputs: torch.Tensor, n_samples: int = 30
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ejecuta N pasadas forward con dropout activo y devuelve media y varianza.

    - mean: probabilidad promedio por clase (prediccion final)
    - variance: varianza entre pasadas (medida de incertidumbre)

    Alta varianza = el modelo duda. Baja varianza = el modelo es consistente.
    """

    enable_dropout_during_inference(model)

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(inputs)
            probabilities = torch.sigmoid(logits)
            samples.append(probabilities)

    # samples: lista de N tensores (batch, n_clases) → apilamos en (N, batch, n_clases)
    stacked = torch.stack(samples, dim=0)

    mean = stacked.mean(dim=0)
    variance = stacked.var(dim=0)

    return mean, variance
