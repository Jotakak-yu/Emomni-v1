#!/bin/bash

# ============================================================
# Emomni Distributed Serving Script
# 一键启动分布式服务架构
# 支持多模型多GPU部署
# ============================================================
#
# 使用方法:
#
# 【单模型启动】
# bash scripts/run_serve.sh --model /path/to/model
#
# 【多模型多GPU启动】
# bash scripts/run_serve.sh --models /path/model1,/path/model2 --gpus 0,1
#
# 【仅启动Controller】
# bash scripts/run_serve.sh --controller-only
#
# 【仅启动单个Worker】
# bash scripts/run_serve.sh --worker-only --model /path/to/model --gpu 0 --port 21002
#
# 【仅启动WebUI】
# bash scripts/run_serve.sh --webui-only
#
# 【停止所有服务】
# bash scripts/run_serve.sh --stop
#
# 【查看实时日志】
# bash scripts/run_serve.sh --logs [controller|worker_0|webui|all]
#
# ============================================================

set -e

# ====== 默认配置 ======
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Conda 环境 (如需自定义，修改此处)
CONDA_ENV="${CONDA_ENV:-emomni}"

# Python 解释器路径 (优先使用 conda 环境)
if [ -f "$HOME/anaconda3/envs/$CONDA_ENV/bin/python" ]; then
    PYTHON="$HOME/anaconda3/envs/$CONDA_ENV/bin/python"
elif [ -f "$HOME/miniconda3/envs/$CONDA_ENV/bin/python" ]; then
    PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
else
    PYTHON="python"
fi

# 确保在正确的 conda 环境中运行
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]; then
    echo "警告: 当前 conda 环境为 $CONDA_DEFAULT_ENV，建议先运行: conda activate $CONDA_ENV"
fi

# 服务配置
CONTROLLER_HOST="0.0.0.0"
CONTROLLER_PORT=21001
WORKER_BASE_PORT=21002
WEBUI_HOST="0.0.0.0"
WEBUI_PORT=7860

# 模型配置 (支持单模型和多模型)
MODEL_PATH=""
MODELS=""  # 逗号分隔的多个模型路径
QWEN_MODEL=""
GPUS=""
USE_EMOTION=true

# TTS配置
ENABLE_TTS=true
TTS_API_URL="http://127.0.0.1:8882"

# 控制选项
CONTROLLER_ONLY=false
WORKER_ONLY=false
WEBUI_ONLY=false
STOP_ALL=false
SINGLE_GPU=""
SINGLE_PORT=""
SHOW_LOGS=""

# PID文件目录
PID_DIR="$PROJECT_DIR/logs/serve/pids"
LOG_DIR="$PROJECT_DIR/logs/serve"

# ====== 使用说明 ======
print_usage() {
    cat << EOF
Emomni 分布式服务启动脚本

用法: $0 [选项]

基本选项:
  --model <path>            单个模型路径
  --models <paths>          多个模型路径，逗号分隔
                            例如: /path/model1,/path/model2
  --gpus <ids>              GPU ID列表，逗号分隔，与模型一一对应
                            例如: 0,1 (model1用GPU0, model2用GPU1)
  --qwen-model <path>       基础Qwen模型路径 (可选)
  --use-emotion             启用情感感知模式 (默认: 开启)
  --no-emotion              禁用情感感知模式

端口配置:
  --controller-port <port>  Controller端口 (默认: 21001)
  --worker-port <port>      Worker基础端口 (默认: 21002)
  --webui-port <port>       WebUI端口 (默认: 7860)

启动模式:
  --controller-only         仅启动Controller
  --worker-only             仅启动单个Worker
  --webui-only              仅启动WebUI
  --gpu <id>                指定单个GPU (用于 --worker-only)
  --port <port>             指定Worker端口 (用于 --worker-only)

TTS配置:
  --enable-tts              启用TTS语音回复 (默认: 开启)
  --no-tts                  禁用TTS语音回复
  --tts-url <url>           TTS API地址 (默认: http://127.0.0.1:8882)

日志与管理:
  --stop                    停止所有服务
  --status                  查看服务状态
  --logs <target>           查看实时日志
                            target: controller, worker_0, worker_1, webui, all

示例:
  # 单GPU启动完整服务
  $0 --model /path/to/model

  # 多GPU启动 (2个Worker)
  $0 --model /path/to/model --num-workers 2 --gpus 0,1

  # 仅启动Controller
  $0 --controller-only

  # 在指定GPU启动Worker
  $0 --worker-only --model /path/to/model --gpu 1

  # 停止所有服务
  $0 --stop

EOF
}

# ====== 解析参数 ======
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --qwen-model)
            QWEN_MODEL="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --gpu)
            SINGLE_GPU="$2"
            shift 2
            ;;
        --port)
            SINGLE_PORT="$2"
            shift 2
            ;;
        --use-emotion)
            USE_EMOTION=true
            shift
            ;;
        --no-emotion)
            USE_EMOTION=false
            shift
            ;;
        --controller-port)
            CONTROLLER_PORT="$2"
            shift 2
            ;;
        --worker-port)
            WORKER_BASE_PORT="$2"
            shift 2
            ;;
        --webui-port)
            WEBUI_PORT="$2"
            shift 2
            ;;
        --controller-only)
            CONTROLLER_ONLY=true
            shift
            ;;
        --worker-only)
            WORKER_ONLY=true
            shift
            ;;
        --webui-only)
            WEBUI_ONLY=true
            shift
            ;;
        --enable-tts)
            ENABLE_TTS=true
            shift
            ;;
        --no-tts)
            ENABLE_TTS=false
            shift
            ;;
        --tts-url)
            TTS_API_URL="$2"
            shift 2
            ;;
        --stop)
            STOP_ALL=true
            shift
            ;;
        --status)
            # 显示服务状态
            echo "=========================================="
            echo "Emomni 服务状态"
            echo "=========================================="
            if [ -d "$PID_DIR" ]; then
                for pid_file in "$PID_DIR"/*.pid; do
                    if [ -f "$pid_file" ]; then
                        name=$(basename "$pid_file" .pid)
                        pid=$(cat "$pid_file")
                        if kill -0 "$pid" 2>/dev/null; then
                            echo "✅ $name (PID: $pid) - 运行中"
                        else
                            echo "❌ $name (PID: $pid) - 已停止"
                        fi
                    fi
                done
            else
                echo "未发现运行中的服务"
            fi
            exit 0
            ;;
        --logs)
            SHOW_LOGS="${2:-all}"
            shift
            if [[ "$SHOW_LOGS" != "controller" && "$SHOW_LOGS" != "webui" && "$SHOW_LOGS" != "all" && ! "$SHOW_LOGS" =~ ^worker_ ]]; then
                shift  # 有参数则多移动一位
            fi
            # 实时查看日志
            echo "=========================================="
            echo "Emomni 实时日志 - $SHOW_LOGS"
            echo "=========================================="
            echo "按 Ctrl+C 退出日志查看"
            echo ""
            case "$SHOW_LOGS" in
                controller)
                    tail -f "$LOG_DIR/controller.log" 2>/dev/null || echo "日志文件不存在: $LOG_DIR/controller.log"
                    ;;
                webui)
                    tail -f "$LOG_DIR/webui.log" 2>/dev/null || echo "日志文件不存在: $LOG_DIR/webui.log"
                    ;;
                worker_*)
                    tail -f "$LOG_DIR/${SHOW_LOGS}.log" 2>/dev/null || echo "日志文件不存在: $LOG_DIR/${SHOW_LOGS}.log"
                    ;;
                all)
                    tail -f "$LOG_DIR"/*.log 2>/dev/null || echo "日志目录为空或不存在: $LOG_DIR/"
                    ;;
                *)
                    echo "未知日志目标: $SHOW_LOGS"
                    echo "可选: controller, worker_0, worker_1, ..., webui, all"
                    exit 1
                    ;;
            esac
            exit 0
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "错误: 未知参数 $1"
            print_usage
            exit 1
            ;;
    esac
done

# ====== 创建必要目录 ======
mkdir -p "$PID_DIR"
mkdir -p "$PROJECT_DIR/logs/serve"

# ====== 停止所有服务 ======
stop_all_services() {
    echo "=========================================="
    echo "停止所有 Emomni 服务"
    echo "=========================================="
    
    if [ -d "$PID_DIR" ]; then
        for pid_file in "$PID_DIR"/*.pid; do
            if [ -f "$pid_file" ]; then
                name=$(basename "$pid_file" .pid)
                pid=$(cat "$pid_file")
                if kill -0 "$pid" 2>/dev/null; then
                    echo "停止 $name (PID: $pid)..."
                    kill "$pid" 2>/dev/null || true
                    sleep 1
                    # 强制终止
                    if kill -0 "$pid" 2>/dev/null; then
                        kill -9 "$pid" 2>/dev/null || true
                    fi
                fi
                rm -f "$pid_file"
            fi
        done
    fi
    
    echo "所有服务已停止"
}

if [ "$STOP_ALL" = true ]; then
    stop_all_services
    exit 0
fi

# ====== 启动Controller ======
start_controller() {
    echo "启动 Controller (端口: $CONTROLLER_PORT)..."
    
    CUDA_VISIBLE_DEVICES="" $PYTHON -m serve.controller \
        --host "$CONTROLLER_HOST" \
        --port "$CONTROLLER_PORT" \
        --dispatch-method shortest_queue \
        > "$PROJECT_DIR/logs/serve/controller.log" 2>&1 &
    
    echo $! > "$PID_DIR/controller.pid"
    echo "Controller 已启动 (PID: $!)"
    sleep 2
}

# ====== 启动Worker ======
start_worker() {
    local gpu_id=$1
    local worker_port=$2
    local worker_id=$3
    local model_path=$4  # 新增：模型路径参数
    
    echo "启动 Worker $worker_id (GPU: $gpu_id, 端口: $worker_port, 模型: $model_path)..."
    
    WORKER_ADDR="http://localhost:$worker_port"
    CONTROLLER_ADDR="http://localhost:$CONTROLLER_PORT"
    
    EMOTION_FLAG=""
    if [ "$USE_EMOTION" = true ]; then
        EMOTION_FLAG="--use-emotion"
    fi
    
    QWEN_FLAG=""
    if [ -n "$QWEN_MODEL" ]; then
        QWEN_FLAG="--qwen-model $QWEN_MODEL"
    fi
    
    CUDA_VISIBLE_DEVICES="$gpu_id" $PYTHON -m serve.model_worker \
        --host "0.0.0.0" \
        --port "$worker_port" \
        --worker-address "$WORKER_ADDR" \
        --controller-address "$CONTROLLER_ADDR" \
        --model-path "$model_path" \
        $QWEN_FLAG \
        $EMOTION_FLAG \
        > "$PROJECT_DIR/logs/serve/worker_${worker_id}.log" 2>&1 &
    
    echo $! > "$PID_DIR/worker_${worker_id}.pid"
    echo "Worker $worker_id 已启动 (PID: $!, GPU: $gpu_id)"
    sleep 3
}

# ====== 启动WebUI ======
start_webui() {
    echo "启动 WebUI (端口: $WEBUI_PORT)..."
    
    CONTROLLER_URL="http://localhost:$CONTROLLER_PORT"
    
    $PYTHON -m serve.gradio_web_server \
        --host "$WEBUI_HOST" \
        --port "$WEBUI_PORT" \
        --controller-url "$CONTROLLER_URL" \
        > "$PROJECT_DIR/logs/serve/webui.log" 2>&1 &
    
    echo $! > "$PID_DIR/webui.pid"
    echo "WebUI 已启动 (PID: $!)"
}

# ====== 主逻辑 ======
cd "$PROJECT_DIR"

echo "=========================================="
echo "Emomni 分布式服务启动"
echo "=========================================="
echo "项目目录: $PROJECT_DIR"
echo ""

# 验证参数
if [ "$CONTROLLER_ONLY" = false ] && [ "$WEBUI_ONLY" = false ]; then
    if [ -z "$MODEL_PATH" ] && [ -z "$MODELS" ]; then
        echo "错误: 必须指定模型路径 (--model 或 --models)"
        print_usage
        exit 1
    fi
    # 单模型验证
    if [ -n "$MODEL_PATH" ] && [ ! -d "$MODEL_PATH" ]; then
        echo "错误: 模型路径不存在: $MODEL_PATH"
        exit 1
    fi
fi

# 解析多模型列表 (如果使用 --models)
if [ -n "$MODELS" ]; then
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
else
    MODEL_ARRAY=("$MODEL_PATH")
fi

# 解析GPU列表
if [ -n "$GPUS" ]; then
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
elif [ -n "$SINGLE_GPU" ]; then
    GPU_ARRAY=("$SINGLE_GPU")
else
    GPU_ARRAY=("0")
fi

# 验证多模型路径
if [ -n "$MODELS" ]; then
    for mp in "${MODEL_ARRAY[@]}"; do
        if [ ! -d "$mp" ]; then
            echo "错误: 模型路径不存在: $mp"
            exit 1
        fi
    done
fi

# 仅启动Controller
if [ "$CONTROLLER_ONLY" = true ]; then
    start_controller
    echo ""
    echo "=========================================="
    echo "Controller 启动完成"
    echo "地址: http://localhost:$CONTROLLER_PORT"
    echo "=========================================="
    exit 0
fi

# 仅启动Worker
if [ "$WORKER_ONLY" = true ]; then
    if [ -z "$SINGLE_GPU" ]; then
        echo "错误: --worker-only 模式需要指定 --gpu"
        exit 1
    fi
    worker_port="${SINGLE_PORT:-$WORKER_BASE_PORT}"
    start_worker "$SINGLE_GPU" "$worker_port" "0" "${MODEL_ARRAY[0]}"
    echo ""
    echo "=========================================="
    echo "Worker 启动完成"
    echo "=========================================="
    exit 0
fi

# 仅启动WebUI
if [ "$WEBUI_ONLY" = true ]; then
    start_webui
    echo ""
    echo "=========================================="
    echo "WebUI 启动完成"
    echo "地址: http://localhost:$WEBUI_PORT"
    echo "=========================================="
    exit 0
fi

# ====== 完整启动流程 ======
NUM_MODELS=${#MODEL_ARRAY[@]}
NUM_GPUS=${#GPU_ARRAY[@]}

# 多模型模式：每个模型对应一个Worker
if [ "$NUM_MODELS" -gt 1 ]; then
    echo "多模型模式:"
    for i in "${!MODEL_ARRAY[@]}"; do
        echo "  模型 $i: ${MODEL_ARRAY[$i]}"
    done
    NUM_WORKERS=$NUM_MODELS
else
    echo "模型路径: ${MODEL_ARRAY[0]}"
    # 单模型模式：默认1个Worker
    NUM_WORKERS=${NUM_WORKERS:-1}
fi

echo "Worker数量: $NUM_WORKERS"
echo "GPU列表: ${GPU_ARRAY[*]}"
echo "情感感知: $USE_EMOTION"
echo "TTS启用: $ENABLE_TTS"
echo ""

# 1. 启动Controller
start_controller

# 2. 启动Workers
for ((i=0; i<NUM_WORKERS; i++)); do
    # 模型：多模型时每个Worker用不同模型，单模型时所有Worker用同一模型
    if [ "$NUM_MODELS" -gt 1 ]; then
        model_path=${MODEL_ARRAY[$i]}
    else
        model_path=${MODEL_ARRAY[0]}
    fi
    
    # GPU：循环使用GPU列表
    gpu_idx=$((i % NUM_GPUS))
    gpu_id=${GPU_ARRAY[$gpu_idx]}
    
    worker_port=$((WORKER_BASE_PORT + i))
    start_worker "$gpu_id" "$worker_port" "$i" "$model_path"
done

# 等待Worker注册
echo "等待Worker注册..."
sleep 5

# 3. 启动WebUI
start_webui

# 等待WebUI启动
sleep 2

echo ""
echo "=========================================="
echo "所有服务启动完成！"
echo "=========================================="
echo ""
echo "📡 Controller: http://localhost:$CONTROLLER_PORT"
for ((i=0; i<NUM_WORKERS; i++)); do
    worker_port=$((WORKER_BASE_PORT + i))
    if [ "$NUM_MODELS" -gt 1 ]; then
        model_name=$(basename "${MODEL_ARRAY[$i]}")
    else
        model_name=$(basename "${MODEL_ARRAY[0]}")
    fi
    echo "🔧 Worker $i:   http://localhost:$worker_port ($model_name)"
done
echo "🌐 WebUI:      http://localhost:$WEBUI_PORT"
echo ""
echo "日志目录: $PROJECT_DIR/logs/serve/"
echo ""
echo "管理命令:"
echo "  停止服务:  bash $0 --stop"
echo "  查看状态:  bash $0 --status"
echo "  查看日志:  bash $0 --logs [controller|worker_0|webui|all]"
echo "=========================================="