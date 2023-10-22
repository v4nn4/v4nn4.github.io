---
layout: post
title: Training LeNet-5 on Armenian script
---

Following [Tinkering with Tesseract]({% post_url 2023-10-13-tinkering-with-tesseract %}), I wanted to gain a better understanding of how OCR systems work. So, I decided to start with building my own character recognition engine using [PyTorch](https://github.com/pytorch/pytorch). The code is available at [v4nn4/hynet](https://github.com/v4nn4/hynet).

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

Since the font has serifs, the difference between capital and small letters is not obvious. Maybe `Օ` and `օ` could be merged into a single character, but I preferred keeping all characters.

For each set of 76 characters, we generate an image consisting of 56x56 pixels and apply 20 rotations, 20 blurs, and 3 mode filters. We generated a total of 91,200 examples, shuffled the dataset, and used an 80% training/test split to avoid overfitting.

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

Here is an example with the letter `խ` as a 56x56 pixels image :
![Network input](/assets/images/network_input.svg){: .center}

Here are 4 batches of the training set :

![Training dataset](/assets/images/training_set.png){: .center}

Next, we implemented LeNet-5 (1998)[^1] using PyTorch. Here is the model definition :

```python
class LeNet(nn.Module):
    """LeNet-5 (1998)"""

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

        # Initialize weights
        gain = nn.init.calculate_gain("tanh")
        nn.init.xavier_normal_(self.conv1.weight, gain)
        nn.init.xavier_normal_(self.conv2.weight, gain)
        nn.init.xavier_normal_(self.fc1.weight, gain)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight, gain)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_normal_(self.fc3.weight, 1.0)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = (x - self.mean) / self.std  # normalize
        x = self.pool(torch.tanh(self.conv1(x)))  # conv2d + tanh + pool
        x = self.pool(torch.tanh(self.conv2(x)))  # conv2d + tanh + pool
        x = x.view(-1, 16 * self.D * self.D)  # flatten
        x = torch.tanh(self.fc1(x))  # fc layer + tanh
        x = torch.tanh(self.fc2(x))  # fc layer + tanh
        x = self.fc3(x)  # fc layer
        return self.sm(x)  # softmax to output probabilities
```

We kept most of the original architecture and applied modern techniques . Some things that improved training :

- **Normalizing data** : we normalize input pixel intensity by the mean and standard deviation observed in the training sample
- **Optimizer** : we use the Adam[^2] optimizer instead of regular SGD, as it is known to yield better results in many cases. We clearly observe a gain in accuracy on our dataset
- **Initialization** : we use normal Xavier[^3] to set our initial weight distribution to help gradient flow
- **Activation function** : even though the ReLU activation function is recommended in general, for this particular problem we found that tanh performs much better


Some techniques that either had no effect or decreased accuracy :
- **Learning rate decay** : we used `torch.optim.lr_scheduler.StepLR` to schedule learning rate decay. It is probably interesting for longer training, but not especially helpful here
- **Early stopping** : it is important to not waste compute resources if we are not to monitor the training live. As we are only using a few epochs here, we can just manually stop our training and adjust the number of epochs
- **Batch normalization** : we tried it after the convolutional layers and after the fully connected layers, but it did not improve accuracy. It is probably because our weight initialization is already good, and the training is short
- **Dropout** : it switches off some layers at random. It did not improve accuracy on the test set

We trained our model for 10 epochs and stopped at epoch 5. We achieved an almost perfect fit, with 99.5% accuracy. The initialization seems to do the trick as we start very high on the first epoch. Here is the training report :

![Training report](/assets/images/report.svg){: .center}

It is interesting to note that we only misclassify a single character in our training set :

```
ր => ը    310 (32%)
```

This was a character that we identified as a potential trouble maker in [Tinkering with Tesseract]({% post_url 2023-10-13-tinkering-with-tesseract %}). While we successfully learned from our training set, this overfitting comes at a cost. Some tests show that our model fails to generalize to other fonts.

The graphs below show, for each image representing each character (no rotation, no blur) the probability output by the network for each predicted character. The one with the highest probability has the largest size. A straight line indicates that the model achieves perfect accuracy.

![Evaluation on Mk_Parz_U-Iatlic](/assets/images/evaluation_Mk_Parz_U-Italic.png){: .center}

When using the regular font Mk_Parz_U, the accuracy drops to 59.2%. In the training set, we used negative rotations that resemble the non-italic font. However, those samples account for a only fraction of the total set, around a minority as we use a set of angles ranging from -40° to 0.

![Evaluation on Mk_Parz_U](/assets/images/evaluation_Mk_Parz_U.png){: .center}

With a sans-serif font like Arial, the accuracy drops to 10.5%... Quite bad. Still better than randomly guessing, which is 1/76, or 1.3%, to be fair.

![Evaluation on Arial](/assets/images/evaluation_arial.png){: .center}

We observe that our model fails to generalize effectively to other fonts. This is not entirely surprising, as the network learns the strokes of a single font. Transitioning from one font to another can be challenging without prior knowledge of the stylistic variations. However, we do notice some degree of generalization when training on an italic font and testing on its regular version. The transformation from regular to italic primarily involves rotating the non-horizontal strokes. Our data augmentation using rotations might have helped, but was not sufficient to reach a good accuracy.

That is all for today ✌.

[^1]: LeCun, Y., Bottou, L., Bengio, Y. & Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE (p./pp. 2278--2324).
[^2]: Kingma, Diederik P., and Jimmy Ba. “Adam: A method for stochastic optimization.” arXiv preprint arXiv:1412.6980 (2014).
[^3]: Glorot, Xavier & Bengio, Y.. (2010). Understanding the difficulty of training deep feedforward neural networks. Journal of Machine Learning Research - Proceedings Track. 9. 249-256.