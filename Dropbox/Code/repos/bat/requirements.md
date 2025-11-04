# This project Demonstrates a bat using relationships to move and reason.

## ⚠️ IMPORTANT: Cloud Instance Execution Required

**All computation must be performed on cloud instances (EC2/Lambda Cloud, vast.ai), not locally.**

### Configuration Files
- **`config.yaml`**: Vast.ai instance connection details
- **`main_macbook.py`**: Automated connection script
- **`cursorrules`**: Development guidelines emphasizing cloud execution

---

## 🚀 QUICK START: Vast.ai Connection (RECOMMENDED)

### Prerequisites
1. **SSH Key**: Ensure `~/.ssh/vastai_key` exists and has correct permissions
   ```bash
   chmod 600 ~/.ssh/vastai_key
   ```

2. **Python 3**: Use `python3` command (not `python`)
   ```bash
   python3 --version  # Should be 3.8+
   ```

3. **Config File**: Verify `config.yaml` exists with correct instance details

4. **Virtual Environment**: Set up and activate virtual environment (see macOS section below)

### Step 0: Activate Virtual Environment (REQUIRED)
```bash
# Navigate to project directory
cd /Users/mike/Dropbox/Code/repos/bat

# Create virtual environment if it doesn't exist (first time only)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Or use the activation script
source activate_venv.sh
```

### Step 1: Test Connection
```bash
# From MacBook, in project directory
cd /Users/mike/Dropbox/Code/repos/bat
python3 main_macbook.py --test
```

### Step 2: Initialize Git Repository (First Time Only)
```bash
# Clone repository on vast.ai instance
python3 main_macbook.py --command "cd /root && rm -rf bat && git clone https://github.com/mkn9/bat.git && cd bat && pwd && git status"
```

### Step 3: Verify Environment
```bash
# Check Python dependencies
python3 main_macbook.py --command "cd /root/bat && python3 -c \"import pandas, numpy, matplotlib, seaborn; print('✅ All packages installed')\""
```

### Step 4: Ready to Use
```bash
# Open interactive shell
python3 main_macbook.py --shell

# Or execute remote commands
python3 main_macbook.py --command "cd /root/bat && python3 your_script.py"
```

**⚠️ CRITICAL NOTES:**
- **Always use `python3`** (not `python`) on macOS
- **Vast.ai user**: `root` (not `ubuntu`)
- **Project path**: `/root/bat` (not `/home/ubuntu/bat`)
- **SSH port**: Check `config.yaml` for current port (usually 40542)

---

### Cloud Instance Setup and Connection


### **🚨 CRITICAL: Git Repository Setup**

**When starting a new cloud instance, you MUST initialize the git repository:**

#### For Vast.ai Instances (Recommended Method)
```bash
# From MacBook, use main_macbook.py script
python3 main_macbook.py --command "cd /root && rm -rf bat && git clone https://github.com/mkn9/bat.git && cd bat && pwd && git status"

# Or manually via SSH
ssh -i ~/.ssh/vastai_key -p 40542 root@[instance-ip]
cd /root
git clone https://github.com/mkn9/bat.git
cd bat
pwd  # Should show: /root/bat
git status  # Should show: On branch main
```

#### For EC2/Lambda Cloud Instances
```bash
# Step 1: Connect to cloud instance
ssh -i ~/.ssh/LambdaKey.pem ubuntu@[instance-ip]

# Step 2: Clone the repository (REQUIRED for new instances)
git clone https://github.com/mkn9/bat.git
cd bat

# Step 3: Verify setup
pwd  # Should show: /home/ubuntu/bat
git status  # Should show: On branch main
ls -la  # Should show all project files
```

**Why Git Setup is Critical:**
- Cloud instances start as clean Ubuntu systems
- No project files exist until repository is cloned
- Git operations will fail without proper repository setup
- All development files must be pulled from GitHub

**Connection Recovery:**
If connection drops, reconnect and ensure you're in the git directory:

**Vast.ai:**
```bash
python3 main_macbook.py --command "cd /root/bat && git pull"
# Or manually:
ssh -i ~/.ssh/vastai_key -p 40542 root@[instance-ip]
cd /root/bat  # Navigate to project directory
git pull   # Get latest changes if needed
```

**EC2/Lambda Cloud:**
```bash
ssh -i ~/.ssh/LambdaKey.pem ubuntu@[instance-ip]
cd ~/bat  # Navigate to project directory
git pull   # Get latest changes if needed
```

### **Step-by-Step Setup Procedure**

#### 1. Initial Connection and Project Transfer
```bash
# Test connection
ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@[instance-ip] "echo 'Connection successful'; uname -a"

# Transfer complete project directory
scp -i /Users/mike/keys/LambdaKey.pem -r /path/to/local/bat ubuntu@[instance-ip]:~/bat

# Verify project structure
ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@[instance-ip] "cd bat && pwd && ls -la"
```

#### 2. Python Package Installation
```bash
# Install missing Python packages
pip3 install pymongo seaborn pdf2image requests --user

# Verify installations
python3 -c "import pandas, numpy, pymongo, matplotlib, seaborn; print('All packages installed successfully')"
```


#### 5. Final Environment Verification
```bash
# Complete dependency check
python3 -c "
import pandas, numpy, matplotlib, seaborn, requests
print(f'✅ pandas: {pandas.__version__}')
print(f'✅ numpy: {numpy.__version__}')
print(f'✅ pymongo: {pymongo.__version__}')
print(f'✅ matplotlib: {matplotlib.__version__}')
print(f'✅ seaborn: {seaborn.__version__}')
"

# System tools verification
mongod --version | head -1
```

### **Environment Status (Verified September 2025)**
#### complete as needed

### **Troubleshooting Notes**

#### Package Installation Location
- Python packages installed with `--user` flag to avoid permission issues
- Packages installed in `/home/ubuntu/.local/lib/python3.10/site-packages/`
- Warning about `/home/ubuntu/.local/bin` not in PATH (can be ignored for this project)

#### Git Repository Access
- Private repository requires SSH key or personal access token
- **Alternative**: Transfer complete project via SCP (method used)
- Maintains all git history and configuration

## Legacy System Requirements

### Hardware Requirements
- Minimum 2GB RAM
- 1GB free disk space

## Software Dependencies

### Core Dependencies
- Python 3.8 or newer
- MongoDB (for document storage)
- SQLite3 (for relational data)

### Python Packages
- pdf2image
- pandas
- numpy
- pillow
- requests

## Installation Instructions

### Ubuntu/Debian (EC2 Instance)

1. **Install system dependencies:**
   ```bash
   sudo apt update
   sudo apt install < any required packages>
   ```

2. **Install Python packages:**
   ```bash
   pip install pandas numpy 
   ```

### macOS (Development Only - No Package Installation)

**Note**: Do not install packages on macOS. Use only for code editing and file management.

**⚠️ CRITICAL: Use Virtual Environment to Prevent Accidental Package Installation**

To isolate the MacBook from any accidental package installations, always use a virtual environment:

#### Step 1: Create Virtual Environment (First Time Only)
```bash
# Navigate to project directory
cd /Users/mike/Dropbox/Code/repos/bat

# Create virtual environment in project directory
python3 -m venv venv

# Verify virtual environment was created
ls -la venv/
```

#### Step 2: Activate Virtual Environment
```bash
# Activate virtual environment
source venv/bin/activate

# Verify activation (prompt should show (venv))
which python3  # Should show: .../bat/venv/bin/python3
python3 --version  # Should show Python 3.x
```

#### Step 3: Install Minimal Dependencies (Optional)
Only install packages needed for MacBook scripts (like `pyyaml` for `main_macbook.py`):
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install only essential packages for MacBook operations
pip install pyyaml

# Verify installation
python3 -c "import yaml; print('✅ yaml installed')"
```

#### Step 4: Use Activation Script (Recommended)
```bash
# Use the provided activation script
source activate_venv.sh

# Or manually activate
source venv/bin/activate
```

#### Deactivate Virtual Environment
```bash
# When done working
deactivate
```

**Important Notes:**
- **Always activate virtual environment** before running any Python scripts on MacBook
- **Never install packages globally** on macOS - use virtual environment only
- **Virtual environment isolates** MacBook from accidental installations
- **All computation packages** (pandas, numpy, torch, etc.) should only be installed on vast.ai instance
- **Virtual environment is git-ignored** - each developer creates their own

## Project Structure

## Expected Outputs

### Quality Validation Outputs
- Comprehensive unit test results showing data quality metrics

## Performance Notes

## Troubleshooting

### Connection Issues
- **"Connection refused" or "Connection timed out"**: 
  - **Check SSH port**: Verify port in `config.yaml` matches vast.ai dashboard (usually 40542)
  - **Check SSH key**: Ensure `~/.ssh/vastai_key` exists and has correct permissions (`chmod 600 ~/.ssh/vastai_key`)
  - **Check IP address**: Verify `public_ip` in `config.yaml` matches current vast.ai instance IP
  - **Test manually**: `ssh -i ~/.ssh/vastai_key -p 40542 root@[instance-ip] "echo 'test'"`
- **"Permission denied (publickey)"**: 
  - **Solution**: Verify SSH key path in `config.yaml` is correct (`~/.ssh/vastai_key`)
  - **Check key permissions**: `chmod 600 ~/.ssh/vastai_key`
  - **Verify key exists**: `ls -la ~/.ssh/vastai_key`

### Python Command Issues (CRITICAL)
- **"SyntaxError: invalid syntax" when running main_macbook.py**: Using `python` instead of `python3`
  - **Error**: `python main_macbook.py --test` fails with syntax error
  - **Solution**: Always use `python3 main_macbook.py` on macOS
  - **Why**: macOS may have Python 2.7 as default `python`, but code requires Python 3
  - **Verification**: `python3 --version` should show 3.8 or higher

### Git Repository Issues (CRITICAL)
- **"fatal: not a git repository"**: You're on a cloud instance without the repository cloned
  - **Vast.ai Solution**: `python3 main_macbook.py --command "cd /root && git clone https://github.com/mkn9/bat.git && cd bat && git status"`
  - **EC2/Lambda Solution**: `git clone https://github.com/mkn9/bat.git && cd bat`
  - **Verification**: `git status` should show "On branch main"
- **Git operations fail on cloud instance**: Repository not initialized
  - **Root Cause**: Cloud instances start clean, no project files exist
  - **Prevention**: Always run git clone as first step on new instances
- **Wrong directory for git operations**: 
  - **Vast.ai**: Ensure you're in `/root/bat`
    - **Check**: `pwd` should show `/root/bat`
    - **Fix**: `cd /root/bat`
  - **EC2/Lambda**: Ensure you're in `/home/ubuntu/bat`
    - **Check**: `pwd` should show `/home/ubuntu/bat`
    - **Fix**: `cd ~/bat` or `cd /home/ubuntu/bat`
- **"Permission denied (publickey)" when pushing from MacBook**: SSH not configured for GitHub
  - **Solution**: See "GitHub SSH Setup for MacBook" section above
  - **Quick fix**: Configure `~/.ssh/config` to use `~/.ssh/github_mkn9` key for GitHub

### Development Workflow Issues
- **Context confusion**: Unclear whether you're on MacBook or cloud instance
  - **MacBook**: Use for git operations, file editing, development
  - **Cloud Instance**: Use for computation, script execution only
  - **Check terminal prompt**: `ubuntu@ip-address` = cloud, `mike@MacBook` = local
- **File synchronization**: Changes made on one system not available on another
  - **Solution**: Use git workflow: MacBook → git push → cloud instance git pull

## Security Notes

- SSH key file must have restricted permissions (400 or 600)
- Keep security group rules as restrictive as possible
- Regularly update the allowed IP address in security groups
- Never commit SSH keys to version control

## GitHub SSH Setup for MacBook

### SSH Key Location
The GitHub SSH keys are located at:
- **Private key**: `~/.ssh/github_mkn9`
- **Public key**: `~/.ssh/github_mkn9.pub`

### Initial SSH Configuration

To enable GitHub access from the MacBook, configure SSH to use the GitHub-specific key:

```bash
# Add GitHub host configuration to ~/.ssh/config
cat >> ~/.ssh/config << 'EOF'
Host github.com
  HostName github.com
  IdentityFile ~/.ssh/github_mkn9
  User git
EOF
```

### Verify SSH Configuration

After configuration, test the connection:

```bash
# Test GitHub SSH connection
ssh -T git@github.com

# Expected output:
# Hi mkn9! You've successfully authenticated, but GitHub does not provide shell access.
```

### Setting Up a New Repository

When setting up a new repository to push to GitHub:

```bash
# 1. Add remote origin (replace with your repository URL)
git remote add origin git@github.com:mkn9/[repository-name].git

# 2. Ensure branch is named 'main'
git branch -M main

# 3. Push to GitHub
git push -u origin main
```

### Updating Existing Remote

If the remote URL needs to be changed:

```bash
# Remove existing origin
git remote remove origin

# Add new origin
git remote add origin git@github.com:mkn9/[repository-name].git

# Push to new remote
git push -u origin main
```

### Troubleshooting GitHub SSH Issues

#### "Permission denied (publickey)" Error
- **Check**: SSH config is properly set up
  ```bash
  cat ~/.ssh/config | grep -A 3 "Host github.com"
  ```
- **Verify**: Key file exists and has correct permissions
  ```bash
  ls -la ~/.ssh/github_mkn9*
  chmod 600 ~/.ssh/github_mkn9
  chmod 644 ~/.ssh/github_mkn9.pub
  ```
- **Test**: SSH connection directly
  ```bash
  ssh -T git@github.com
  ```

#### "Repository not found" Error
- Repository doesn't exist on GitHub or you don't have access
- Create the repository on GitHub first, or check repository name/access permissions

#### SSH Config File Missing or Deleted
If `~/.ssh/config` is accidentally deleted, recreate it:

```bash
# Recreate SSH config with GitHub configuration
cat >> ~/.ssh/config << 'EOF'
Host github.com
  HostName github.com
  IdentityFile ~/.ssh/github_mkn9
  User git
EOF

# Verify configuration
ssh -T git@github.com
```

### Copying GitHub SSH Keys to Cloud Instances

When setting up GitHub access on cloud instances (for reference):

```bash
# Copy private key to cloud instance
scp -i ~/.ssh/vastai_key -P [PORT] ~/.ssh/github_mkn9 root@[IP_ADDRESS]:~/.ssh/id_ed25519

# Copy public key to cloud instance
scp -i ~/.ssh/vastai_key -P [PORT] ~/.ssh/github_mkn9.pub root@[IP_ADDRESS]:~/.ssh/id_ed25519.pub

# Set proper permissions on cloud instance
ssh -i ~/.ssh/vastai_key -p [PORT] root@[IP_ADDRESS] "chmod 600 ~/.ssh/id_ed25519 && chmod 644 ~/.ssh/id_ed25519.pub"

# Test GitHub connection from cloud instance
ssh -i ~/.ssh/vastai_key -p [PORT] root@[IP_ADDRESS] "ssh -T git@github.com"
```

**Note**: Replace `[PORT]` and `[IP_ADDRESS]` with your actual cloud instance details.

## Chat History Management Process

### Overview
This project maintains comprehensive chat history records to track all interactions, decisions, and development progress. All conversations between users and AI assistants are systematically recorded and version controlled.

### Directory Structure
```
chat_history/
├── README.md                           # Directory documentation
├── chat_history_template.md            # Template for new sessions
├── session_YYYYMMDD_HHMMSS.md         # Individual session files
├── current_session.md                  # Active/ongoing session
└── sessions_index.md                   # Index of all sessions
```

### Process Instructions

#### 1. Starting a New Chat Session
```bash
# Copy template to create new session file
cp chat_history/chat_history_template.md chat_history/current_session.md

# Edit session information
# - Update date, time, session ID
# - Set session objectives
# - Initialize participant information
```

#### 2. Recording Chat Interactions
**For each user input and AI response:**

1. **Timestamp each interaction** with `[HH:MM]` format
2. **Copy complete messages** including:
   - All user text/queries
   - Complete AI responses
   - Code blocks with proper formatting
   - Tool calls and results
   - File modifications

3. **Use proper markdown formatting**:
   ```markdown
   ### [14:30] User Input:
   ```
   [User's complete message]
   ```

   ### [14:31] AI Response:
   ```
   [AI's complete response including code blocks]
   ```
   ```

#### 3. Documenting Key Information
Throughout the session, maintain:

- **Key Decisions Made**: Important choices and reasoning
- **Files Created/Modified**: Complete list with descriptions
- **Action Items**: Tasks and follow-up items
- **Code Changes Summary**: Brief overview of modifications
- **Session Outcomes**: What was accomplished

#### 4. Ending a Chat Session
```bash
# 1. Complete the session summary sections
# 2. Set session status to "Completed"
# 3. Update timestamp
# 4. Rename file with session timestamp
mv chat_history/current_session.md chat_history/session_$(date +%Y%m%d_%H%M%S).md

# 5. Update sessions index
# Add entry to chat_history/sessions_index.md with session summary
```

#### 5. Version Control Integration
```bash
# Add chat history files to git
git add chat_history/

# Commit with descriptive message
git commit -m "Chat history: [Brief session description]

- Session objectives and outcomes
- Key files modified
- Major decisions made"

# Push to remote repository
git push origin main
```
