import torch
from torch import nn


class LinearActivationsPruner(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        sparsity_type=None,
        sparsity_ratio=None,
        name=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_type = sparsity_type
        self.sparsity_ratio = sparsity_ratio
        self.name = name

        # Исправляем инициализацию weight - должен быть правильной формы
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))

        # Инициализируем bias правильно
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def activations_pruner(self, x, sparsity_ratio):
        """Pruning активаций по порогу"""
        if sparsity_ratio <= 0.0:
            return x  # Нет pruning
        if sparsity_ratio >= 1.0:
            return torch.zeros_like(x)  # Полное обнуление

        abs_values = torch.abs(x)
        batch_size, seq_len, features = x.shape

        # Reshape для правильной работы с батчами
        abs_values_flat = abs_values.view(-1, features)
        x_flat = x.view(-1, features)

        # Вычисляем количество элементов для обнуления (k наименьших)
        k = int(sparsity_ratio * features)
        if k > 0 and k < features:
            # Находим k-й наименьший элемент для каждого токена
            threshold_values = torch.kthvalue(abs_values_flat, k, dim=1, keepdim=True)[0]
            # Маска: True для элементов, которые нужно оставить (>= threshold)
            mask = abs_values_flat >= threshold_values
            x_flat = x_flat * mask.float()

        return x_flat.view(batch_size, seq_len, features)

    def forward(self, x):
        # Применяем pruning активаций только для режима masked-activations-layer
        if self.sparsity_type == "masked-activations-layer":
            x = self.activations_pruner(x, self.sparsity_ratio)

        # Выполняем линейное преобразование
        output = torch.matmul(x, self.weight.t())

        if self.bias is not None:
            output = output + self.bias

        return output

    def set_sparsity_ratio(self, sparsity_ratio):
        self.sparsity_ratio = sparsity_ratio

    @classmethod
    def from_original(
        cls,
        orig_linear,
        sparsity_type=None,
        sparsity_ratio=None,
        name=None
    ):
        """Создает LinearActivationsPruner из существующего nn.Linear слоя"""
        linear_sp = cls(
            orig_linear.in_features,
            orig_linear.out_features,
            bias=orig_linear.bias is not None,
            sparsity_type=sparsity_type,
            sparsity_ratio=sparsity_ratio,
            name=name
        )

        # Копируем веса из оригинального слоя
        linear_sp.weight.data.copy_(orig_linear.weight.data)

        if orig_linear.bias is not None and linear_sp.bias is not None:
            linear_sp.bias.data.copy_(orig_linear.bias.data)

        return linear_sp


def replace_linears_with_pruner(module, sparsity_ratio):
    """
    Рекурсивная функция для замены nn.Linear слоев на LinearActivationsPruner

    Args:
        module: Модуль PyTorch для обработки
        sparsity_ratio: Коэффициент разреженности активаций
    """
    for name, child in module.named_children():
        # Если нашли обычный Linear — создаём pruner из него
        if isinstance(child, torch.nn.Linear):
            pruner = LinearActivationsPruner.from_original(
                child,
                sparsity_type="masked-activations-layer",
                sparsity_ratio=sparsity_ratio,
                name=name
            ).to(child.weight.device)
            setattr(module, name, pruner)
        elif isinstance(child, LinearActivationsPruner):
            child.set_sparsity_ratio(sparsity_ratio)
        else:
            # Иначе рекурсивно смотрим глубже
            replace_linears_with_pruner(child, sparsity_ratio)