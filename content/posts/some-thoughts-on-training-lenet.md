---
author: "v4nn4"
title: "Some thoughts on training LeNet"
date: "2023-12-30"
tags: ["cnn", "lenet", "armenian"]
ShowToc: false
ShowBreadCrumbs: false
math: mathjax
---

Since my last blog post [Training LeNet on Armenian script]({{< ref "posts/training-lenet-on-armenian-script.md" >}} "Training LeNet on Armenian script"), I have made some significant improvement to the training process.

## Model simplification

To reduce the number of hyperparameters, I hardcoded the mean and standard deviation used for normalizing pixel intensities. For this, I arbitrarily decided that a square occupying one-third of the total pixel space would represent an average character.

The mean and standard deviation of the pixel intensities are respectively $1/3$ (a third is black) and $\sqrt{2}/3$. Those values are close to those observed on the training set.

I also chose to focus only on the lower case letters for this project to reduce the number of classes to 38.

## Accuracy metrics

Upon inspecting my code, I realized that certain PyTorch functions perform more tasks than initially anticipated. For instance, `torch.nn.CrossEntropyLoss` accepts integer labels (indices), eliminating the need for one-hot encoding. While this might be convenient in some cases, I believe one-hot encoding the labels for a classification task is significantly more readable.

Here is my current training loop, with one-hot encoded labels and accuracy score calculated by `torchmetrics`:

```python
for epoch in epochs:

    train_loss, train_acc = 0, 0
    
    for inputs, labels in train_dataloader:

        # Evaluate logits and loss
        logits = model(inputs)
        loss = criterion(logits, labels)

        # Compute metrics
        train_loss += loss.item()
        max_indices = torch.argmax(logits, dim=1)
        preds = F.one_hot(max_indices, num_classes=logits.size(1)).float()
        train_acc += multiclass_exact_match(
            preds=preds, target=labels, num_classes=num_classes
        )

        # Gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
```

## Extensive logging

I also started logging weight and biases in [TensorBoard](https://www.tensorflow.org/tensorboard) to check their stability. It has become a standard and is very easy to use:

```python
writer = SummaryWriter(log_dir=log_dir)

for epoch in epochs:
    
    # train...
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
```

Additionally, I added an ASCII logger for my character images which helped me realize that Arial did not support Armenian and was generating the unknown character symbol □. Here is an example of a rotated and noised ն character:

```verbatim
7   7  7 7      7 77 7  7  7  7 
  7 7 77  7   77      77777 77  
7777   777  8166077777   77  77 
    7 7 77 510000000 7  77 7 7 7
 77 7  77 750000000000 7   777 7
  7   77 710000000000006 7      
7  7      10000134400006  77 7  
7  7 7    000005  74045777   7 7
7 7777  700000997   7 7    77   
77  77 7700000677 2 7  777777   
7 77 77 00000097  60477  7  77  
 7  77  00000977 7000408 777   7
 7    77000009 7700000067 7    7
7    7 0000009 7730000677 77 77 
 77  7 000009 7  300006  77 777 
  7777 00000 7  00000037 7      
777 770000006   300006   7 777 7
777 77000009 7733000037777 7 77 
   77 100006 77000002    77    7
   7 2000059   0000037      7777
7777 4000057  3000002777  7  777
 77  000000  5000003 77 7 7 7 7 
 77 7300001101000002      77 7 7
 7 77640000000000002 77   7  77 
 77 77 0000000000027    777 7 7 
77777 7334155400006 7 7  7  7 7 
7   7 7 77  70000267 7  7   77  
7 777  77 7 77 003       7      
  7 7 77 77777 7    777  77  77 
 7777 777  77  777  7    7  7777
   7      7777 77 7   77    7  7
7 7 77777   777 7      7 777 7    
```

## Bonus

Some additional findings:

- Notebooks are excellent for visualizing data and analyzing results. However, I found that a robust CLI for preparing, training, and evaluating the model is even more beneficial. I have been using [fire](https://github.com/google/python-fire), which automatically generates a CLI from Python classes
- `tbparse` helps extract data from TensorBoard as pandas DataFrame for plotting
- [Hugging Face](https://huggingface.co/) provides [Gradio](https://www.gradio.app/) app hosting to run AI models. Once the model weights are saved, the model can be easily instantiated from a Gradio app and used for inference. You can find the space [here](https://huggingface.co/spaces/romflorentz/armenian-ocr)
