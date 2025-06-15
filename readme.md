# ConversationTTS: A Speech Foundation Model for Multilingual Conversational Text-to-Speech

![ConversationTTS](https://img.shields.io/badge/Release-Download%20Latest%20Version-blue?style=for-the-badge&logo=github)

## Introduction

Welcome to the ConversationTTS repository. This project provides the training and inference code for a state-of-the-art speech synthesis model. Our model is designed for multilingual conversational text-to-speech applications. 

We are excited to share the first checkpoint, which has been trained on 1.5 epochs using approximately 200,000 hours of speech data. This foundational model can serve various applications, from voice assistants to educational tools.

### Checkpoint V1: 1B-20w-1.5epoch

To get started, you can download the initial checkpoint using the following command:

```bash
wget https://huggingface.co/AudioFoundation/SpeechFoundation/resolve/main/ckpt1.checkpoint
```

This checkpoint provides a solid base for further experimentation and development.

### Data

The training process utilized a diverse set of large-scale TTS datasets, including:

- **Emili-Yodas**
- **WenetSpeech**
- **MLS**
- **People Speech**

In addition to these datasets, we collected a variety of podcast datasets in multiple languages, including English, Chinese, and Cantonese. We label different speakers in the datasets using identifiers like [1], [2], and so on. 

Currently, the model is trained on 200,000 hours of data. Future updates will include checkpoints trained on over 500,000 hours of data, enhancing the model's capabilities and performance.

## Usage

### ⚡ Quick Start

To quickly set up and run ConversationTTS, follow the instructions below.

1. **Clone the Repository:**

   First, clone the repository to your local machine:

   ```bash
   git clone https://github.com/Rapinto1712/ConversationTTS.git
   cd ConversationTTS
   ```

2. **Install Dependencies:**

   Ensure you have all the required libraries installed. You can do this using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Checkpoint:**

   Use the command provided earlier to download the checkpoint. 

4. **Run the Model:**

   After downloading the checkpoint, you can run the model using the provided scripts. For detailed instructions, refer to the [Installation & Usage Instructions](docs/quick_use.md).

### 🛠️ Local Deployment

If you prefer to run ConversationTTS locally, follow these steps:

1. **Install the Required Software:**

   Make sure you have Python 3.7 or later installed. You will also need to install additional dependencies specified in the `requirements.txt` file.

2. **Set Up the Environment:**

   Create a virtual environment for better dependency management:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Run CapSpeech Locally:**

   After setting up the environment and installing the dependencies, you can run the CapSpeech application. For more details, check the [Installation & Usage Instructions](docs/quick_use.md).

## Development

We welcome contributions to improve ConversationTTS. Please refer to the following documentation for development guidelines and best practices.

### Contributing

To contribute to this project, please fork the repository and create a new branch for your feature or bug fix. Once your changes are ready, submit a pull request for review.

### Issues

If you encounter any issues while using ConversationTTS, please check the [Issues](https://github.com/Rapinto1712/ConversationTTS/issues) section of the repository. You can also submit new issues if you find bugs or have feature requests.

## Future Work

We plan to expand the capabilities of ConversationTTS by:

- Training on larger datasets.
- Improving the multilingual capabilities.
- Enhancing the voice quality and naturalness of the speech output.

Stay tuned for updates and new releases.

## Releases

For the latest updates and downloadable checkpoints, please visit our [Releases](https://github.com/Rapinto1712/ConversationTTS/releases) section. 

You can find the latest checkpoint and additional resources there.

## Conclusion

Thank you for your interest in ConversationTTS. We hope this model serves as a useful tool for your projects. If you have any questions or feedback, feel free to reach out through the Issues section of this repository. 

For more information and updates, please check our [Releases](https://github.com/Rapinto1712/ConversationTTS/releases). 

We look forward to seeing how you use ConversationTTS in your applications!