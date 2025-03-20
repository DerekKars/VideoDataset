# pytorch-VideoDataset
Tools for loading video dataset and transforms on video in pytorch. You can directly load video files without preprocessing.

## Requirements

+ pytorch
+ torchvision
+ numpy
+ python-opencv
+ PIL

## How to use

1. Place the files [datasets.py](./datasets.py) and [transforms.py](./transforms.py) at your project directory.

2. Create csv file to declare where your video data are. The format of your csv file should like:

   ```csv
   path
   ~/path/to/video/file1.mp4
   ~/path/to/video/file2.mp4
   ~/path/to/video/file3.mp4
   ~/path/to/video/file4.mp4
   ```

   if the videos of your dataset are saved as image in folders. The format of your csv file should like:

   ``` 
   path
   ~/path/to/video/folder1/
   ~/path/to/video/folder2/
   ~/path/to/video/folder3/
   ~/path/to/video/folder4/
   ```

3. Prepare video datasets and load video to `torch.Tensor`.

   ```python
   import torch
   import torchvision
   import datasets
   import transforms
   
   dataset = datasets.VideoDataset(
   	"./data/example_video_file.csv",
       transform=torchvision.transforms.Compose([
           transforms.VideoFilePathToTensor(max_len=50, fps=10, padding_mode='last'),
           transforms.VideoRandomCrop([512, 512]),
           transforms.VideoResize([256, 256]),
       ])
   )
   data_loader = torch.utils.data.DataLoader(dataset, batch_size = 2, shuffle = True)
   for videos in data_loader:
       print(videos.size())
   ```

   If the videos of your dataset are saved as image in folders. You can use `VideoFolderPathToTensor` transfoms rather than `VideoFilePathToTensor` .

   ```python
   import torch
   import torchvision
   import datasets
   import transforms
   
   dataset = datasets.VideoDataset(
   	"./data/example_video_folder.csv",
       transform=torchvision.transforms.Compose([
           transforms.VideoFolderPathToTensor(max_len=50, padding_mode='last'),
           transforms.VideoRandomCrop([512, 512]),
           transforms.VideoResize([256, 256]),
       ])
   )
   data_loader = torch.utils.data.DataLoader(dataset, batch_size = 2, shuffle = True)
   for videos in data_loader:
       print(videos.size())
   ```

4. You can use `VideoLabelDataset` to load both video and label.

   ```python
   import torch
   import torchvision
   import datasets
   import transforms
   
   dataset = datasets.VideoLabelDataset(
   	"./data/example_video_file_with_label.csv",
       transform=torchvision.transforms.Compose([
           transforms.VideoFilePathToTensor(max_len=50, fps=10, padding_mode='last'),
           transforms.VideoRandomCrop([512, 512]),
           transforms.VideoResize([256, 256]),
       ])
   )
   data_loader = torch.utils.data.DataLoader(dataset, batch_size = 2, shuffle = True)
   for videos, labels in data_loader:
       print(videos.size(), labels)
   ```

5. You can also customize your dataset. It's easy to create your own `CustomVideoDataset` class and reuse the transforms I provided to transform video path to `torch.Tensor` and do some preprocessing such as `VideoRandomCrop`. 

   

## Docs

### [datasets](./datasets.py)

+ #### **datasets.VideoDataset**

  Video Dataset for loading video. 

  It will output only path of video (neither video file path or video folder path). However, you can load video as torch.Tensor (C x L x H x W). See below for an example of how to read video as torch.Tensor. Your video dataset can be image frames or video files.

  + **Parameters**

    + **csv_file** (str): path fo csv file which store path of video file or video folder. The format of csv_file should like:

      ```csv
      # example_video_file.csv   (if the videos of dataset is saved as video file)
      
      path
      ~/path/to/video/file1.mp4
      ~/path/to/video/file2.mp4
      ~/path/to/video/file3.mp4
      ~/path/to/video/file4.mp4
      
      # example_video_folder.csv   (if the videos of dataset is saved as image frames)
      
      path
      ~/path/to/video/folder1/
      ~/path/to/video/folder2/
      ~/path/to/video/folder3/
      ~/path/to/video/folder4/
      ```

  + **Example**

    if the videos of dataset is saved as video file.

    ```python
    import torch
    from datasets import VideoDataset
    import transforms
    dataset = VideoDataset(
        "example_video_file.csv",
        transform = transforms.VideoFilePathToTensor()  # See more options at transforms.py
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)
    for videos in data_loader:
    	print(videos.size())
    ```

    if the video of dataset is saved as frames in video folder. The tree like: (The names of the images are arranged in ascending order of frames)

    ```shell
    ~/path/to/video/folder1
    ├── frame-001.jpg
    ├── frame-002.jpg
    ├── frame-003.jpg
    └── frame-004.jpg
    ```

    ```python
    import torch
    from datasets import VideoDataset
    import transforms
    dataset = VideoDataset(
        "example_video_folder.csv",
        transform = transforms.VideoFolderPathToTensor()  # See more options at transforms.py
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)
    for videos in data_loader:
    	print(videos.size())
    ```

+ #### **datasets.VideoLabelDataset**

  Dataset Class for Loading Video with label.

  It will output path and label. However, you can load video as torch.Tensor (C x L x H x W). See below for an example of how to read video as torch.Tensor.

  You can load tensor from video file or video folder by using the same way as VideoDataset.

  + **Parameters**

    + **csv_file** (str): path fo csv file which store path and label of video file (or video folder). The format of csv_file should like:

      ```csv
      path, label
      ~/path/to/video/file1.mp4, 0
      ~/path/to/video/file2.mp4, 1
      ~/path/to/video/file3.mp4, 0
      ~/path/to/video/file4.mp4, 2
      ```

  + **Example**

    ```python
    import torch
    import transforms
    dataset = VideoDataset(
        "example_video_file_with_label.csv",
        transform = transforms.VideoFilePathToTensor()  # See more options at transforms.py
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)
    for videos, labels in data_loader:
        print(videos.size())
    ```

### [transforms](./transforms.py)

All transforms at here can be composed with `torchvision.transforms.Compose()`.

+ #### **transforms.VideoFilePathToTensor** 

  load video at given file path to torch.Tensor (C x L x H x W, C = 3). 

  + **Parameters**
    + **max_len** (int): Maximum output time depth (L <= max_len). Default is None. If it is set to None, it will output all frames. 
    + **fps** (int): sample frame per seconds. It must lower than or equal the origin video fps. Defaults to None. 
    + **padding_mode** (str): Type of padding. Default to None. Only available when max_len is not None.
      + None: won't padding, video length is variable.
      + 'zero': padding the rest empty frames to zeros.
      + 'last': padding the rest empty frames to the last frame.

+ #### **transforms.VideoFolderPathToTensor**

  load video at given folder path to torch.Tensor (C x L x H x W).

  + **Parameters**
    + **max_len** (int): Maximum output time depth (L <= max_len). Default is None. If it is set to None, it will output all frames. 
    + **padding_mode** (str): Type of padding. Default to None. Only available when max_len is not None.
      + None: won't padding, video length is variable.
      + 'zero': padding the rest empty frames to zeros.
      + 'last': padding the rest empty frames to the last frame.

+ #### **transforms.VideoResize**

  resize video tensor (C x L x H x W) to (C x L x h x w).

  + **Parameters**
    + **size** (sequence): Desired output size. size is a sequence like (H, W), output size will matched to this.
    + **interpolation** (int, optional): Desired interpolation. Default is `PIL.Image.BILINEAR`

+ #### **transforms.VideoRandomCrop**

  Crop the given Video Tensor (C x L x H x W) at a random location.

  + **Parameters**
    + **size** (sequence): Desired output size like (h, w).

+ #### **transforms.VideoCenterCrop**

  Crops the given video tensor (C x L x H x W) at the center.

  + **Parameters**
    + **size** (sequence): Desired output size of the crop like (h, w).

+ #### **transforms.VideoRandomHorizontalFlip**

  Horizontal flip the given video tensor (C x L x H x W) randomly with a given probability.

  + **Parameters**
    + **p** (float): probability of the video being flipped. Default value is 0.5.

+ #### **transforms.VideoRandomVerticalFlip**

  Vertical flip the given video tensor (C x L x H x W) randomly with a given probability.

  + **Parameters**
    + **p** (float): probability of the video being flipped. Default value is 0.5.

+ #### **transforms.VideoGrayscale**

  Convert video (C x L x H x W) to grayscale (C' x L x H x W, C' = 1 or 3)

  + **Parameters**
    + **num_output_channels** (int): (1 or 3) number of channels desired for output video.
   
   ### Video Classification Datasets

#### 1. ‌**UCF101**‌

- ‌**Size**‌: 13,320 video clips
- ‌**Categories**‌: 101 human action classes (e.g., diving, playing guitar), covering human-object interactions, sports, and diverse scenarios.
- ‌**Features**‌: High diversity in camera motion, lighting, and backgrounds; widely used for action recognition and cross-scene classification.

#### 2. ‌**HMDB51**‌

- ‌**Size**‌: 6,766 video clips
- ‌**Categories**‌: 51 fine-grained human motion classes (e.g., brushing hair, fencing), including facial expressions, body movements, and multi-person interactions.
- ‌**Features**‌: Sourced from movies and YouTube; rich in motion details for studying human dynamics and local feature modeling.

#### 3. ‌**Kinetics-600**‌

- ‌**Size**‌: Approximately 650,000 videos
- ‌**Categories**‌: 600 diverse action classes (e.g., playing instruments, repairing cars), covering daily life and domain-specific activities.
- ‌**Features**‌: Large-scale, high-quality annotations; ideal for pretraining video models or evaluating classification under long-tailed distributions.

#### 4. ‌**YouTube-8M**‌

- ‌**Size**‌: 6.1 million videos (preprocessed into 1.9 billion frames)
- ‌**Categories**‌: 3,800+ multi-label visual entities (e.g., animals, scenes, activities) annotated via YouTube’s knowledge graph.
- ‌**Features**‌: Supports multi-label classification and cross-modal analysis; suitable for large-scale content understanding and topic mining.

#### 5. ‌**anetvideos**‌

- ‌**Size**‌: 1 000 videos
- ‌**Categories**‌: YouTube’s: animals, scenes, activities.
- ‌**Features**‌: Suitable for large-scale content topic mining.

#### 6. ‌**AI Challenger**‌

- ‌**Size**‌: 200,000 short videos
- ‌**Categories**‌: 63 trending elements (e.g., dance, fitness) with multi-label annotations (subject, scene, action).
- ‌**Features**‌: Focuses on vertical mobile videos and user-generated content (UGC), including edited clips with effects; practical for real-world applications.

------

### Video Topic Analysis Datasets

#### 1. ‌**MSR-VTT**‌

- ‌**Size**‌: 10,000 video clips
- ‌**Categories**‌: 20 general topics (e.g., music, travel) with natural language descriptions.
- ‌**Features**‌: Cross-modal data for video-to-text mapping; supports topic annotation and content summarization.

#### 2. ‌**ActivityNet**‌

- ‌**Size**‌: 20,000 videos
- ‌**Categories**‌: 200 complex activities (e.g., walking a dog, assembling furniture) with temporal boundary annotations.
- ‌**Features**‌: Dense event annotations in long videos; suitable for temporal topic segmentation and hierarchical semantic analysis.

#### 3. ‌**Charades**‌

- ‌**Size**‌: 9,800 videos
- ‌**Categories**‌: 157 indoor daily activities (e.g., closing windows, making beds) with multi-label annotations.
- ‌**Features**‌: Realistic home scenarios with object interactions; ideal for fine-grained topic modeling.

#### 4. ‌**TikTok Dataset**‌

- ‌**Size**‌: Not specified (public dataset on Kaggle)
- ‌**Categories**‌: User-generated short videos (e.g., dance, comedy clips) with metadata and engagement metrics.
- ‌**Features**‌: Reflects social media trends; useful for analyzing cultural topics and user behavior patterns.

#### 5. ‌**Something-Something V2**‌

- ‌**Size**‌: 220,000 videos
- ‌**Categories**‌: 174 object interaction classes (e.g., pushing left, tearing objects) emphasizing temporal causality.
- ‌**Features**‌: Context-dependent action definitions; designed for studying topic evolution and logical reasoning in dynamic scenes.

------

### Video Clustering Datasets

#### 1. ‌**Sports-1M**‌

- ‌**Size**‌: 1.1 million videos
- ‌**Categories**‌: 487 sports-related activities (e.g., soccer, skiing) with partial automated annotations.
- ‌**Features**‌: Large-scale but sparsely labeled; suitable for unsupervised clustering or semi-supervised scene discovery.

#### 2. ‌**AVA (Atomic Visual Actions)**‌

- ‌**Size**‌: 430 movie clips (1.7 million action annotations)
- ‌**Categories**‌: 80 atomic actions (e.g., walking, shaking hands) with spatiotemporal localization.
- ‌**Features**‌: Dense annotations in multi-person scenes; supports clustering based on action similarity.

#### 3. ‌**places3**v‌

- ‌**Size**‌: 10,000 videos selected from different video sites（Selected from place3）
- ‌**Categories**‌:  Multiple types and multiple scenes.
- ‌**Features**‌: Easy to integrate video categories.

#### 4. ‌**ASLAN**‌

- ‌**Size**‌: 3,697 video pairs
- ‌**Categories**‌: 432 action similarity labels (e.g., "opening a door" vs. "closing a drawer").
- ‌**Features**‌: Focuses on action similarity rather than class labels; ideal for metric learning and cross-category clustering.

#### 5. ‌**videoc-cs**‌

- ‌**Size**‌: 500 videos
- ‌**Categories**‌:  Objects, people change.
- ‌**Features**‌: Focus on style classification research.

#### 6. ‌**HMDB51-Motion**‌

- ‌**Size**‌: 6,766 videos (shared with HMDB51)
- ‌**Categories**‌: 51 motion patterns (e.g., hand trajectories, limb speed) extracted via optical flow or keypoints.
- ‌**Features**‌: Captures dynamic motion features for grouping videos based on movement patterns.

#### 7. ‌**GER-vid**‌

- ‌**Size**‌: Not specified (public dataset of model videos)
- ‌**Categories**‌: Fashion-related actions (e.g., runway walking, outfit changes) with pose and apparel annotations.
- ‌**Features**‌: High visual diversity; suitable for identity- or style-agnostic visual clustering.

