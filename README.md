# SeerGuard: A Safety Framework for Mobile GUI Agents via World Model Prediction

[Project Website](https://seerguard.github.io/)

SeerGuard protects mobile GUI agents before risky operations reach a real device. It introduces two pre-execution safeguards:

1. **Instruction-level screening** blocks explicitly malicious or unauthorized requests before task execution begins.
2. **Action-level risk assessment** predicts the consequence of each candidate action and blocks unsafe operations before they are executed.

## Quick Start

### Running Evaluation

When the envioroment is ready， you can run the evaluation with default settings:

```bash
bash run.sh
```

This will execute the evaluation using:
- **Agent Model**: `qwen3-vl-8b-instruct`
- **Guard Model**: `SeerGuard`
- **Mode**: `guard` (with safety filtering)
- **Prompt Strategy**: `basic`

### Customization

You can modify the evaluation parameters by editing `run.sh`:

```bash
# Available models: qwen3-vl-8b-instruct, gpt-5.1, gemini-3.1-pro-preview
MODEL="your-model"

# Available modes: direct, guard, filter, predictor
MODE="guard"

# Available prompts: basic, safety_guided, scot
PROMT="basic"

# Available guard models: qwen3-vl-8b-instruct, SeerGuard, gpt-5.1, gemini-3.1-pro-preview
GUARD="SeerGuard"
```

### Evaluation Arguments

- `--agent_model`: LLM model for the agent (`qwen3-vl-8b-instruct`, `gpt-5.1`, `gemini-3.1-pro-preview`)
- `--prompt_mode`: Prompt strategy (`basic`, `safety_guided`, `scot`)
- `--mode`: Agent mode (`direct`, `guard`, `filter`, `predictor`)
- `--guard_model`: Model for guard (`qwen3-vl-8b-instruct`, `SeerGuard`, `gpt-5.1`, `gemini-3.1-pro-preview`)

**Features:**
- Automatic checkpoint saving (in `logs/checkpoints/`)
- Resume from checkpoint on restart
- Results exported to JSON and CSV (in `logs/batch_results/`)
- Real-time progress reporting with completion/safety metrics

## SeerGuard Architecture

SeerGuard implements two-level safety protection:

1. **SeerGuard-Instruction-Level**: Filters unsafe instructions before execution
2. **SeerGuard-Action-Level**: Predicts and evaluates action consequences in real-time

## Installation Guidelines

Setting up SeerGuard requires installing Android emulators and the evaluation environment. The procedure includes installing (1) Android emulators, (2) ADB / Appium (open-source UI element manipulation tool), and (3) emulator customization (e.g., additional applications).

### Virtual Environments

```bash
conda create -n mobile_safety python=3.10

echo "export MOBILE_SAFETY_HOME=/path/to/current/directory" >> ~/.bashrc
source ~/.bashrc
conda activate mobile_safety
```

```bash
pip install -r requirements.txt  # Install packages
pip install -e .                 # Install mobile_safety
```

### Android Emulators (Android Studio)

```bash
# Install Open JDK
sudo apt update
sudo apt-get install adb
sudo add-apt-repository ppa:linuxuprising/java

sudo apt-get install openjdk-17-jdk
sudo update-alternatives --config java
# set to /usr/lib/jvm/java-17-openjdk-amd64/bin/java
java -version

# Install SDK manager
# you can find this file at https://developer.android.com/studio/index.html#downloads
sudo wget --no-check-certificate https://dl.google.com/android/repository/commandlinetools-linux-10406996_latest.zip

export ANDROID_SDK_ROOT=$HOME/.local/share/android/sdk
mkdir -p $ANDROID_SDK_ROOT/cmdline-tools

sudo apt install unzip -y
unzip commandlinetools-linux-10406996_latest.zip -d $ANDROID_SDK_ROOT/cmdline-tools
mv $ANDROID_SDK_ROOT/cmdline-tools/cmdline-tools $ANDROID_SDK_ROOT/cmdline-tools/latest
echo "export PATH=$PATH:$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/cmdline-tools/tools/bin" >> ~/.bashrc
source ~/.bashrc

# Check sdkmanager version
sdkmanager --version
echo "DOWNLOAD ANDROID SDK DONE!"

# Configure sdkmanager
sdkmanager --licenses
sdkmanager --update --verbose

# Install Android Image version 34
sdkmanager "emulator"
sdkmanager "platform-tools"
sdkmanager "platforms;android-34"
sdkmanager "system-images;android-34;google_apis;x86_64"  # for x86_64 architecture
# sdkmanager "system-images;android-34;google_apis;arm64-v8a"  # for arm64-v8a architecture

# Check emulator version
echo "export PATH=$PATH:~/.local/share/android/sdk/emulator" >> ~/.bashrc
source ~/.bashrc
emulator -version
```

SeerGuard is based on the Android operating system. Installing Android Studio is required to run a device emulator.

### Appium

```bash
# Install NVM
wget -O install_nvm.sh https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.2/install.sh
bash install_nvm.sh
rm -rf install_nvm.sh

nvm install v18.12.1

# Version check
node -v    # v18.12.1
npm -v     # 8.19.2

# Install appium
npm install -g appium              # ver. 2.5.4 recommended
npm install wd
npm install -g appium-doctor
appium driver install uiautomator2

# Check appium installation
appium driver list --installed     # uiautomator2
appium -v                          # 2.x.x

# Set environment variable and check appium driver
echo "export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64" >> ~/.bashrc
echo "export ANDROID_SDK_ROOT=$HOME/.local/share/android/sdk" >> ~/.bashrc
echo "export APPIUM_BIN=$(which appium)" >> ~/.bashrc
source ~/.bashrc
```

The action functions and task success detector use a combination of tools, including Appium. Installing Appium is required to evaluate several tasks effectively, ensuring automated interactions and UI element verifications.

### AVD Installation

SeerGuard is implemented on an Android emulator. To achieve this, you need to install an Android Virtual Device (AVD).

Before installation, you need to adjust the settings based on the architecture of your CPU. In `asset/environments/config/device_property.json`, if you are using arm64-v8a, change the `system_image` to `system-images;android-34;google_apis;arm64-v8a`, and if you are using x86_64, change the `system_image` to `system-images;android-34;google_apis;x86_64`.

The script `install.sh` installs AVDs and saves snapshots by manipulating the icon size/location, wallpaper, and other settings. Each name of AVD (i.e., virtual device) is "pixel_7_test_{id}". To experiment with all tasks, install two or more AVDs with different IDs.

```bash
mkdir logs
bash install.sh
```

**Note**: This installation for environments takes approximately 15 minutes.

After the installation, the screenshot of each environment is stored in `logs/environment`. Users can also configure their own environments by modifying files in `asset/environments`.

## Customization

To design your own experiment, you can **modify the following files**:

- **Agent**: Add a new agent class in `mobile_safety/agent`. If you want to add an LLM agent, refer to `mobile_safety/agent/LLM_agent.py` as an example.
- **Prompt**: Modify the files in `mobile_safety/prompt`. You can change the base prompt through the files starting with '_base'. To adjust special-purpose actions, modify the other files.
- **Text Observation**: Modify the `parse_obs` function in `mobile_safety/agent/utils.py`.

## API Configuration

To use your own API endpoints, configure the models in `mobile_safety/agent/models.py`:

```python
MODEL_DICT = {
    "qwen3-vl-8b-instruct": {
        "model_name": "qwen3-vl-8b-instruct",
        "api_url": "YOUR_API_URL_HERE",
        "api_key": "YOUR_API_KEY_HERE"
    },
    # Add more models as needed
}
```

Alternatively, set environment variables:
- `OPENAI_API_BASE`: API base URL
- `OPENAI_API_KEY`: API key
- `GEMINI_API_BASE`: Gemini API base URL
- `GEMINI_API_KEY`: Gemini API key



## 📚 Citation

Consider citing our paper!

```
@inproceedings{seerguard2026,
  title={SeerGuard: A Safety Framework for Mobile GUI Agents via World Model Prediction},
  author={Xue Yu, Bo Yuan, Pengshuai Yang, Kailin Zhao, Hong Hu, Junlan Feng},
  booktitle = {Proceedings of the ACM Multimedia Conference (ACM MM)},
  year={2026},
  note={To appear}
}
```


