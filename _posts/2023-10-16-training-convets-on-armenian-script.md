---
layout: post
title: Training ConvNets on Armenian script
---

Following [Tinkering with Tesseract]({% post_url 2023-10-13-tinkering-with-tesseract %}), I wanted to get a better understanding of how OCR system work. So I decided to start with building my own character recognition engine using [pytorch](https://github.com/pytorch/pytorch). The code is available at [v4nn4/hynet](https://github.com/v4nn4/hynet).

First, we visualize the alphabet in our target font, [Mk_Parz_U-Italic](https://fonter.am/en/fonts/mk-parz-unicode) :

```python
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

caps = range(0x531, 0x557)
smalls = range(0x561, 0x588)
letters = [f"{chr(a)}{chr(b)}" for (a, b) in zip(caps, smalls)]
letters = [" ".join(letters[i : i + 10]) + "\n" for i in range(0, len(letters), 10)]
letters = " ".join(letters)
# Աա Բբ Գգ Դդ Եե Զզ Էէ Ըը Թթ Ժժ
# Իի Լլ Խխ Ծծ Կկ Հհ Ձձ Ղղ Ճճ Մմ
# Յյ Նն Շշ Ոո Չչ Պպ Ջջ Ռռ Սս Վվ
# Տտ Րր Ցց Ււ Փփ Քք Օօ Ֆֆ

image = Image.new("L", size=(800, 225), color=255)
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("Mk_Parz_U-Italic.ttf", 51)
draw.text((10, 0), alphabet, font=font, fill=0, align="center")
plt.imshow(image, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.gcf().set_size_inches(10, 3)
```

![Mk_Park_U](/assets/images/alphabet.png){: .center}

Beautiful. Since it has serifs, the difference between capital and small letters is less obvious.

For each 76 character, we generate images of size 56x56 pixels and apply 20 rotations, 20 blurs and 3 mode filters. We generate a total of 91,200 examples. We shuffle the dataset and use a 80% training/test split to avoid overfitting.

```python
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch

N = 56  # 56x56 pixels
character = "խ"
image = Image.new("L", size=(N, N), color=255)
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("Mk_Parz_U-Italic.ttf", 51)
x0, y0, x1, y1 = font.getbbox(character)
draw.text(
    (-x0 + (N - (x1 - x0)) / 2, -y0 + (N - (y1 - y0)) / 2),  # center
    character,
    font=font,
    fill=0,
)
image = image.rotate(5, fillcolor=255)  # default 0
image = image.filter(ImageFilter.ModeFilter(2))  # default 0
image = image.filter(ImageFilter.BoxBlur(0.5))  # default 0
pixel_values = list(image.getdata())
T = torch.tensor(pixel_values, dtype=torch.float32) / 255.0
T = T.view((1, N, N))  # network input
```

Example with the letter խ as a 56x56 pixels image :
![Network input](/assets/images/network_input.svg){: .center}

Here are 4 batches of the training set :

![Training dataset](/assets/images/training_set.png){: .center}

We then implement LeNet-5 (1998)[^1] with torch. We kept most of the original architecture. Some key features that helped training :

- **Normalizing data** : we normalize input pixel intensity by the mean and standard deviation observed on the training sample
- **Optimizer** : we use the Adam optimizer instead of regular SGD, as it is known to yield better results in many cases. We clearly observe a gain in accuracy on our dataset
- **Initialization** : we use normal Xavier to set our initial weight distribution to help gradient flow


Some cool training features ✨ that did not help :
- **Learning rate decay** : used `torch.optim.lr_scheduler.StepLR` to schedule learning rate decay. Probably interesting for longer training, not especially helpful here
- **Early stopping** : important to not waste compute resources if we are not to monitor the training live. As we are only using a few epochs here, we can just manually kill our training and adjust the number of epochs
- **Batch normalization** : we tried after the convolutional layers and after the fully connected layers, but it did not improve convergence. It is probably because our weight initialization is already good, and we use a small number of epochs

Here is the model definition in torch :

```python
class LeNet(nn.Module):
    """LeNet-5 (modified)"""

    def __init__(self, N: int, C: int, mean: float = 0.5, std: float = 0.01):
        super(LeNet, self).__init__()
        self.mean = mean
        self.std = std
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.D = int(((N - 4) / 2 - 4) / 2)  # 4 for N = 28
        self.fc1 = nn.Linear(16 * self.D * self.D, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, C)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = (x - self.mean) / self.std  # normalize
        x = self.pool(torch.tanh(self.conv1(x)))  # conv2d + tanh + pool
        x = self.pool(torch.tanh(self.conv2(x)))  # conv2d + tanh + pool 
        x = x.view(-1, 16 * self.D * self.D)  # flatten
        x = torch.tanh(self.fc1(x))  # fc layer + tanh
        x = torch.tanh(self.fc2(x))  # fc layer + tanh
        x = self.fc3(x)  # fc layer
        return self.sm(x)  # softmax to output probabilities


def initialize_weights(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("tanh"))
```

We achieve almost perfect fit, above 99% accuracy :

![Training report](/assets/images/report.svg){: .center}

Here are the misclassifications made by the model :

```
ր => ը    19
```

Definitely overfitting our dataset. As a reminder, our model has 251,636 parameters and our dataset 91,200 examples.

The graphs below shows for each character the probability output by the network for each predicted character. The one with the highest probability has the biggest size. If the model was perfect, it would output a straight line.

![Evaluation on Mk_Parz_U-Iatlic](/assets/images/evaluation_Mk_Parz_U-Italic.png){: .center}

Let's see how the font Mk_Parz_U fares when we evaluate it on our trained model.

![Evaluation on Mk_Parz_U](/assets/images/evaluation_Mk_Parz_U.png){: .center}

What about a sans-serif font like Arial ?

![Evaluation on Arial](/assets/images/evaluation_Arial.png){: .center}

We see that our model fails to generalize to other fonts.

Some ideas on how to improve genealization by augmenting the dataset :

- Add noise to increase robustness to change in nearby pixels
- Instead on relying on rotations, add more fonts and their italic/bold variations

That is all for today ✌.

[^1]: LeCun, Y., Bottou, L., Bengio, Y. & Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE (p./pp. 2278--2324)