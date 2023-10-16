---
layout: post
title: Armenian OCR demystified
---

I have recently been experimenting with [Tesseract](https://github.com/tesseract-ocr/tesseract), an Optical Character Recognition (OCR) engine made by Google.

My main goal was to extract text from scans of a 1920s Armenian newspaper and run search queries on it. Terms like *պատերազմ* (war) or *Ֆրանսիա* (France) for instance are likely to be found within the document.

![Armenian newspaper scan (1925)](/assets/images/haratch_1925_08.png){: .center}

Some initial observations on the document :

- **Image segmentation** : there are a lot of different text blocks in the raw document, and it might be challenging to tell them apart. We will focus on a isolated portion of the text to start
- **Pre-processing** : we will have to convert the image to black and white, increase contrast and use denoising to get rid of the verso background noise
- **Font** : the font used in this 1925 newspaper is likely unknown to Tesseract. However, according to [tesseract-ocr/langdata/issues/67](https://github.com/tesseract-ocr/langdata/issues/67), the training data used a dozen fonts with italic and bold variations, so it should not be an issue

## Using Tesseract

In this section, we investigate the Tesseract performance on our sample. We are using Tesseract 5.3.3 version on Windows ([download page](https://github.com/UB-Mannheim/tesseract/wiki)). The [pytesseract](https://github.com/madmaze/pytesseract) module provides a python layer to call the Tesseract executables.


We will use the following sample from the first page of the newspaper as a benchmark :

![Sample](/assets/images/sample.jpg)

I wrote down the *ground truth* by hand :

<div class="message">
Մեը հայրենակիցնեըէն շատերը շփոթութեան մէջ են ինքնու թեան թուղթերու, անցագրի ևն. մասին, մանաւանդ որ վերջերս նոր կարգագրութիւններ եղան:
</div>

which translates to (Google translate):

<div class="message">
Many of our compatriots are confused about their own papers, passports, etc. about, especially since there were new regulations recently.
</div>

Not an Armenian speaker, but this looks plausible given the historical context.

### Pre-processing

We try some brightness and contrast adjustments and measure the [Character Error Rate](https://torchmetrics.readthedocs.io/en/stable/text/char_error_rate.html) (CER). CER measures how wrong our prediction is character by character, accounting for substitutions, insertions and deletions. The lower the better.

```python
from PIL import Image, ImageEnhance
import pytesseract
from torchmetrics.text.cer import CharErrorRate

# Adjust image and call Tesseract
image = Image.open("sample.png").convert("L")  # grayscale
image = ImageEnhance.Contrast(image).enhance(2.5)
image = ImageEnhance.Brightness(image).enhance(2.5)
text = pytesseract.image_to_string(image, lang="hye")

# Flatten into a single string before computing CER
predictions = " ".join(text.splitlines())
ground_truth = " ".join(open("sample.gt.txt", "r").read().splitlines())
cer = CharErrorRate()(predictions, ground_truth).item()
```

We see that contrast and brightness brings CER down to around 10-15%. The variance observed in the CER might just be due to the small size of the dataset.

![Character Error Rate](/assets/images/cer.svg){: .center }

Increased brightness helps getting rid of the verso background noise. However if pushed too high, some shapes start disappearing, and Tesseract might confuse characters. This is especially true as some characters only differ by a single stroke.  For now we will settle on brightness 2.5 and contrast 2.5, but we keep in mind that this might be an overfit.

Sample image with contrast 2.5 and brightness 1.0, 2.5 and 3.0 :

![Sample image contrast 2.5 brightness 1.0](/assets/images/sample_enhanced_b1.0_c2.5.png)
![Sample image contrast 2.5 brightness 2.5](/assets/images/sample_enhanced_b2.5_c2.5.png)
![Sample image contrast 2.5 brightness 3.0](/assets/images/sample_enhanced_b3.0_c2.5.png)

### Results

The text extracted by Tesseract for our chosen image adjustment is

<div class="message">
Բեր ձայրենակիցներեն չատերը շփոխութետն մէջ էն ինքնու թեան թԹուղն երու, անցագրի են. մասին, մանաւանդ որ վերջերս նոր կարդադրությիւններ եղան:
</div>

The CER is 13%, which is an character-level accuracy of 87%. Diffing the two texts (with [difflib](https://docs.python.org/3/library/difflib.html)) line by line shows what went wrong :

```diff
- Մեը հայրենակիցնեըէն շատերը շփոթութեան մէջ են ինքնու
? ^ ^ ^           ^^  ^         ^    ^      ^
+ Բեր ձայրենակիցներեն չատերը շփոխութետն մէջ էն ինքնու
? ^ ^ ^           ^^  ^         ^    ^      ^
- թեան թուղթերու, անցագրի ևն. մասին, մանաւանդ որ վերջերս նոր
?          ^              ^
+ թեան թԹուղն երու, անցագրի են. մասին, մանաւանդ որ վերջերս նոր
?       +   ^^              ^
- կարգագրութիւններ եղան:
?    ^ ^
+ կարդադրությիւններ եղան:
?    ^ ^    +
```

We can see that some errors are made when characters differ only by a single stroke, like with ե and է or with ր and ը. The translation is obviously very far from the translated ground truth.

One thing I realized also is that serifs on է and ր can vary depending on the font. Courier New (serif) has no lower stroke on է, but has lower strokes on ր. The newspaper font has strong lower strokes on ր, such that ր and ը are hard to tell apart. The fonts used to train Tesseract seem to be equally weighted between serif and sans serif fonts. Training on a single font matching closely the one from the newspaper could yield better result.

I tried to train Tesseract on the font [MK Parz (Unicode)](https://fonter.am/en/fonts/mk-parz-unicode), which resembles the one from the newspaper. I used `text2image` to generate training examples (images and ground truth) from a Wikipedia article written in Armenian, but eventually ran into trouble with `tesstrain`. Might train again in the future.

## Armenian character recognition using CNN

In order to get a better understanding of how OCR works, let's build our own character recognition engine using [pytorch](https://github.com/pytorch/pytorch).

First, we visualize the alphabet in our target font :

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

For each 76 character, we generate images of size 56x56 pixels and apply rotations, blurring and mode filters. We also use the regular and italic font to generate a total of 45600 examples. We shuffle the dataset and use a 80/20 training/test split to avoid overfitting.

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

Example with the letter խ
![Network input](/assets/images/network_input.svg){: .center}

Here are 4 batches of the training sample :

![Training dataset](/assets/images/training_set.png){: .center}

We then implement LeNet-5 (1998) in pytorch. Here C designates the channel number (1 since grayscale) and D is the reduced dimension after convolution and pooling. We use normal Xavier initialization to set the weights.

```python
class LeNet(nn.Module):
    """LeNet-5 (1998)"""

    def __init__(self, N: int, C: int):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # image dimension reduces with convolution kernels and pooling
        self.D = int(((N - 4) / 2 - 4) / 2)  # 4 for N = 28
        self.fc1 = nn.Linear(16 * self.D * self.D, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, C)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.D * self.D)  # flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.lsm(x)

def initialize_weights(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(module.weight, gain=gain)
```
