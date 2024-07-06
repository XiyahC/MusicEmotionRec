# MusicEmotionRec
Music Emotion Recognition using RNNs for Data Mining Research Project.

This study explores the application of neural network technologies to recognize emotions conveyed
in music, aiming to enhance music recommendation systems and support therapeutic interventions by
tailoring music to fit listeners’ emotional states. We utilize Russell’s Emotion Quadrant to categorize
music into four distinct emotional regions and develop models capable of accurately predicting these
categories. Our approach involves extracting a comprehensive set of audio features using Librosa
and applying various recurrent neural network architectures, including standard RNNs, Bidirectional
RNNs, and Long Short-Term Memory (LSTM) networks. Initial experiments are conducted using a
dataset of 900 audio clips, labeled according to the emotional quadrants. We compare the performance
of our neural network models against a set of baseline classifiers and analyze their effectiveness in
capturing the temporal dynamics inherent in musical expression. The results indicate that simpler
RNN architectures may perform comparably or even superiorly to more complex models, particularly
in smaller datasets. This research not only enhances our understanding of the emotional impact
of music but also demonstrates the potential of neural networks in creating more personalized and
emotionally resonant music recommendation and therapy systems.

## Datasets

900 30-sec Audio Clips with 4Q Labels(Russell's Emotion Quadrant):
https://mir.dei.uc.pt/downloads.html
https://github.com/XiyahC/MusicEmotionRec/blob/master/README.md
14,000 Audio Dataset (mtg-jamendo-dataset):
https://github.com/MTG/mtg-jamendo-dataset#downloading-the-dataset

## Related Works

Paper Reports: https://arxiv.org/abs/2405.06747

Code documents: Included in this Github Repo.

## Worth to Mention

*This is aimed at recording all possible issues which may exist in a project. Try to avoid them in the future!*

1. It’s always important to check the input and output of the models you plan to use. 

We initially planned to use XLNet for our project, but it typically processes text-based inputs such as sentences or words, rather than pure audio. Although our audio clips may contain lyrics, the features extracted by Librosa are predominantly audio-based. Faced with this, we considered two options: firstly, converting our numerical features into word embeddings to fit XLNet, which is time-consuming and potentially inaccurate; secondly, switching to a different model better suited for audio data. RNNs are well-regarded for music emotion recognition tasks and would allow us to utilize our data more effectively. Given our time constraints, we opted for the second solution, which ensures more stable and interpretable results.

2. Though the datasets are small, it is useful to have an evaluation set.

The evaluation set will help a lot during the fine-tuning, which you give you a basic idea on how to choose the best stopping point.

3. While constructing Neural Networks, need to be careful on nn.Dropout() and nn.BatchNorm()’s positions.

Adding Dropout can prevent overfitting, and normalizing the data batch-by-batch can make sure the data during training process are stable. nn.Dropout() is usually used after activation functions and nn.BatchNorm() is usually used before activation functions. When using nn.Dropout() and nn.BatchNorm() together, it is customary to apply nn.BatchNorm() before nn.Dropout(). This order is preferred because normalizing the data batch-by-batch stabilizes it first, and then dropout is applied to randomly omit some of the data points. This sequence ensures that the data normalization process is not affected by the dropout operation.

4. Facing gradient explosion and vanishing.

There are many ways to deal with gradient explosion and vanishing, we used clipping here to prevent this problem happens. In both RNNs and BRNNs we added gradient clipping. Since LSTM already can prevent gradient explosion and vanishing, we decided not to use gradient clipping in that model.

5. Always store the data for steps.

First, this will save memories, secondly, if the memory explode and the computer crushes, you won’t run all the things from the very beginning.
