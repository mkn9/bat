# SSH Disconnection Handling

## Overview

This document explains how the system handles SSH disconnections during training and checkpoint operations.

## Background Training (✅ Survives SSH Disconnection)

### How It Works

When training is started with `--run-magvit-kinematics`, the training process is launched using:

1. **`nohup`**: Prevents the process from being killed when the terminal session ends
2. **`setsid`**: Creates a new session, detaching from the controlling terminal
3. **Background execution**: Process runs independently of SSH session

```bash
setsid nohup python3 run_training.py > training.log 2>&1 < /dev/null &
```

### What Continues Working

✅ **Training continues**: Model training progresses normally
✅ **Checkpoint saving**: Checkpoints are saved every 5 epochs
✅ **Visualization generation**: PNG images are created every 5 epochs
✅ **Log file writing**: Training progress is written to `training.log`

### What Requires SSH Connection

❌ **Download checkpoint images**: Requires active SSH connection
❌ **Check training status**: Requires active SSH connection
❌ **View logs in real-time**: Requires active SSH connection

## Checkpoint Image Download

### Command: `--download-checkpoint-images`

```bash
python3 main_macbook.py --download-checkpoint-images
```

### Behavior

- **Requires SSH connection**: This command uses `rsync` over SSH
- **If SSH is down**: Command will fail with connection error
- **Training still continues**: Images are still being generated on the server
- **Can retry later**: Once SSH is reconnected, you can download all accumulated images

### When to Use

1. **During training**: Check progress periodically (every few hours)
2. **After SSH reconnection**: Download all accumulated images
3. **Before inference**: Verify training completed successfully

## Check Training Status

### Command: `--check-training-status`

```bash
python3 main_macbook.py --check-training-status
```

### Behavior

- **Requires SSH connection**: Checks process status via SSH
- **Non-interfering**: Safe to run while training is active
- **If SSH is down**: Command will fail, but training continues

## SSH Reconnection

### What Happens When You Reconnect

1. **Training is still running**: Process continues in background
2. **Checkpoints are saved**: All checkpoints are available
3. **Logs are available**: Training log shows full history
4. **Images are generated**: All visualization PNGs are on server

### Recovery Steps

1. **Reconnect SSH**:
   ```bash
   ssh -i ~/.ssh/vastai_key -p 40542 root@174.78.228.101
   ```

2. **Check training status**:
   ```bash
   python3 main_macbook.py --check-training-status
   ```

3. **Download images**:
   ```bash
   python3 main_macbook.py --download-checkpoint-images
   ```

4. **View logs**:
   ```bash
   python3 main_macbook.py --command "tail -50 /root/bat/magvit/experiments/kinematics_magvit/training.log"
   ```

## Verification

### How to Verify Training Continues

1. **Start training**:
   ```bash
   python3 main_macbook.py --run-magvit-kinematics
   ```

2. **Note the PID** (shown in output)

3. **Disconnect SSH** (close terminal or lose connection)

4. **Wait 2-3 hours**

5. **Reconnect and check**:
   ```bash
   python3 main_macbook.py --check-training-status
   ```

6. **Verify checkpoint exists**:
   ```bash
   python3 main_macbook.py --command "ls -lh /root/bat/magvit/experiments/kinematics_magvit/checkpoint*.pth"
   ```

### Expected Results

- ✅ Training process is still running (PID exists)
- ✅ Checkpoint file exists and has grown in size
- ✅ Training log shows continued progress
- ✅ PNG images have been generated

## Troubleshooting

### If Training Stopped

1. **Check if process is running**:
   ```bash
   python3 main_macbook.py --command "ps aux | grep run_training.py"
   ```

2. **Check training log for errors**:
   ```bash
   python3 main_macbook.py --command "tail -100 /root/bat/magvit/experiments/kinematics_magvit/training.log"
   ```

3. **Check disk space**:
   ```bash
   python3 main_macbook.py --command "df -h /root/bat"
   ```

4. **Resume from checkpoint**:
   ```bash
   python3 main_macbook.py --run-magvit-kinematics
   # Training will automatically resume from latest checkpoint
   ```

## Summary

| Operation | Requires SSH | Survives SSH Disconnect |
|-----------|--------------|------------------------|
| Training execution | ❌ No | ✅ Yes (nohup/setsid) |
| Checkpoint saving | ❌ No | ✅ Yes |
| Image generation | ❌ No | ✅ Yes |
| Download images | ✅ Yes | ❌ No (but can retry) |
| Check status | ✅ Yes | ❌ No (but can retry) |
| View logs | ✅ Yes | ❌ No (but can retry) |

**Key Point**: Training continues independently of SSH connection. All monitoring/ downloading operations require SSH but can be done after reconnection.

