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
# bash scripts/run_serve.sh --logs [controller|worker_0|kokoro|webui|all]
#
# ============================================================

set -e
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

# ====== 默认配置 ======
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Respect an explicit interpreter or the activated environment.
PYTHON="${PYTHON:-$(command -v python || command -v python3 || true)}"
if [ -z "$PYTHON" ]; then
    echo "错误: 请先安装 Python 3.11 并激活环境"
    exit 1
fi

# 服务配置
CONTROLLER_HOST="0.0.0.0"
CONTROLLER_PORT=21001
WORKER_BASE_PORT=21002
WEBUI_HOST="0.0.0.0"
WEBUI_PORT=7860

# 模型配置 (支持单模型和多模型)
MODEL_PATH=""
MODEL_SIZE="${MODEL_SIZE:-base}"
MODEL_QUANTIZE="${MODEL_QUANTIZE:-}"
MODEL_PRESET=false
MODELS=""  # 逗号分隔的多个模型路径
QWEN_MODEL=""
GPUS=""
USE_EMOTION=true

# TTS配置
ENABLE_TTS=true
TTS_API_URL="http://127.0.0.1:8880/v1"
TTS_MODE="${TTS_MODE:-Kokoro}"
TTS_GPU="${TTS_GPU:-}"
QUANTIZATION=""
EXTERNAL_TTS=false
CHECK_DEPS=false
SKIP_INSTALL=false
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"

# 控制选项
CONTROLLER_ONLY=false
WORKER_ONLY=false
WEBUI_ONLY=false
STOP_ALL=false
FOREGROUND=false
SINGLE_GPU=""
SINGLE_PORT=""
SHOW_LOGS=""

# PID文件目录
LOG_DIR="${EMOMNI_LOGDIR:-$PROJECT_DIR/logs/serve}"
PID_DIR="${EMOMNI_PID_DIR:-$LOG_DIR/pids}"

# ====== 使用说明 ======
print_usage() {
    cat << EOF
Emomni 分布式服务启动脚本

用法: $0 [选项]

基本选项:
  -s, --size <size>         base / small (默认: base)
  -q, --quantize <bits>     4 (省略时使用原始 BF16 模型)
  --model <path>            自定义模型路径 (与 -s/-q 互斥)
  --models <paths>          多个模型路径，逗号分隔
                            例如: /path/model1,/path/model2
  --gpus <ids>              GPU ID列表，逗号分隔，与模型一一对应
                            例如: 0,1 (model1用GPU0, model2用GPU1)
  --qwen-model <path>       基础Qwen模型路径 (可选)
  --use-emotion             启用情感感知模式 (默认: 开启)
  --no-emotion              禁用情感感知模式
  --load-in-4bit            使用 bitsandbytes 4-bit 加载
  --load-in-8bit            使用 bitsandbytes 8-bit 加载

端口配置:
  --controller-port <port>  Controller端口 (默认: 21001)
  --worker-port <port>      Worker基础端口 (默认: 21002)
  --webui-port <port>       WebUI端口 (默认: 7860)

启动模式:
  --controller-only         仅启动Controller
  --worker-only             仅启动单个Worker
  --webui-only              仅启动WebUI
  --foreground              前台运行完整服务 (用于容器)
  --gpu <id>                指定单个GPU (用于 --worker-only)
  --port <port>             指定Worker端口 (用于 --worker-only)

TTS配置:
  --enable-tts              启用TTS语音回复 (默认: 开启)
  --no-tts                  禁用TTS语音回复
  --tts-url <url>           使用已有外部TTS服务 (自动补齐 /v1)
  --tts-mode <mode>         Kokoro (默认) / CosyVoice3
                            TTS GPU: 环境变量 TTS_GPU (默认: 首个 Worker 的 GPU)
  --check-deps              仅检查所选服务依赖，不安装、不启动
  --no-auto-install         检查依赖但不自动安装

日志与管理:
  --stop                    停止所有服务
  --status                  查看服务状态
  --logs <target>           查看实时日志
                            target: controller, worker_0, worker_1, kokoro, webui, all

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
    case "$1" in
        -s|--size|-q|--quantize|--model|--models|--qwen-model|--num-workers|--gpus|--gpu|--port|--controller-port|--worker-port|--webui-port|--tts-url|--tts-mode)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "错误: $1 需要参数"; exit 1
            fi
            ;;
    esac
    case $1 in
        -s|--size)
            MODEL_SIZE="$2"
            MODEL_PRESET=true
            shift 2
            ;;
        -q|--quantize)
            MODEL_QUANTIZE="$2"
            MODEL_PRESET=true
            shift 2
            ;;
        --foreground)
            FOREGROUND=true
            shift
            ;;
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
        --load-in-4bit|--load-in-8bit)
            QUANTIZATION="$1"
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
            TTS_API_URL="${2%/}"
            EXTERNAL_TTS=true
            shift 2
            ;;
        --tts-mode)
            TTS_MODE="$2"
            shift 2
            ;;
        --check-deps)
            CHECK_DEPS=true
            shift
            ;;
        --no-auto-install)
            SKIP_INSTALL=true
            shift
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
                controller|kokoro|cosyvoice3)
                    tail -f "$LOG_DIR/$SHOW_LOGS.log" 2>/dev/null || echo "日志文件不存在: $LOG_DIR/controller.log"
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
mkdir -p "$LOG_DIR"

is_valid_model_path() {
    local path="$1"
    # 1. 本地目录存在
    if [ -d "$path" ]; then
        return 0
    fi
    # 2. HuggingFace Hub repo_id: 格式为 "namespace/repo-name"
    #    包含 / 但不以 / 或 . 开头，且不含空格
    if [[ "$path" == *"/"* ]] && [[ "$path" != "/"* ]] && [[ "$path" != "./"* ]] && [[ "$path" != "../"* ]] && [[ "$path" != *" "* ]]; then
        return 0
    fi
    return 1
}

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

STARTED=()
cleanup_failed_start() {
    echo "启动失败，停止本次已启动的进程。"
    for name in "${STARTED[@]}"; do
        if [ -f "$PID_DIR/$name.pid" ]; then
            kill "$(cat "$PID_DIR/$name.pid")" 2>/dev/null || true
            rm -f "$PID_DIR/$name.pid"
        fi
    done
}
trap cleanup_failed_start ERR
set -E

# Only probe a live process launched by this script; surface its log on failure.
wait_ready() {
    local name="$1" url="$2"
    local result=0
    "$PYTHON" - "$PID_DIR/$name.pid" "$url" "$STARTUP_TIMEOUT" <<'PYWAIT' || result=$?
import os, sys, time, urllib.request
pid = int(open(sys.argv[1]).read())
deadline = time.monotonic() + int(sys.argv[3])
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        break
    try:
        with urllib.request.urlopen(sys.argv[2], timeout=2) as response:
            if response.status == 200:
                sys.exit(0)
    except Exception:
        time.sleep(1)
sys.exit(1)
PYWAIT
    if [ "$result" -ne 0 ]; then
        echo "错误: $name 启动失败或超时，日志: $LOG_DIR/$name.log"
        tail -n 30 "$LOG_DIR/$name.log"
        return 1
    fi
}

check_not_running() {
    if [ -f "$PID_DIR/$1.pid" ] && kill -0 "$(cat "$PID_DIR/$1.pid")" 2>/dev/null; then
        echo "错误: $1 已运行，请先使用 --stop"
        return 1
    fi
    "$PYTHON" - "$2" <<'PYPORT'
import socket, sys
with socket.socket() as sock:
    try:
        sock.bind(('0.0.0.0', int(sys.argv[1])))
    except OSError:
        sys.exit(f"错误: 端口 {sys.argv[1]} 已占用")
PYPORT
}

start_tts() {
    if [ "$ENABLE_TTS" = false ] || [ "$EXTERNAL_TTS" = true ]; then return; fi
    local backend=kokoro
    if [ "$TTS_MODE" = CosyVoice3 ]; then backend=cosyvoice3; fi
    check_not_running "$backend" 8880
    echo "启动 $TTS_MODE (首次下载模型请稍候)..."
    nohup env CUDA_VISIBLE_DEVICES="$TTS_GPU" "$PYTHON" -m serve.tts.server --backend "$backend" \
        > "$LOG_DIR/$backend.log" 2>&1 < /dev/null &
    echo $! > "$PID_DIR/$backend.pid"
    STARTED+=("$backend")
    wait_ready "$backend" "http://127.0.0.1:8880/health"
}

# ====== 启动Controller ======
start_controller() {
    echo "启动 Controller (端口: $CONTROLLER_PORT)..."
    
    check_not_running controller "$CONTROLLER_PORT"
    nohup env CUDA_VISIBLE_DEVICES="" "$PYTHON" -m serve.controller \
        --host "$CONTROLLER_HOST" \
        --port "$CONTROLLER_PORT" \
        --dispatch-method shortest_queue \
        > "$LOG_DIR/controller.log" 2>&1 < /dev/null &
    
    echo $! > "$PID_DIR/controller.pid"
    STARTED+=("controller")
    echo "Controller 已启动 (PID: $!)"
    wait_ready controller "http://127.0.0.1:$CONTROLLER_PORT/health"
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
    
    local QWEN_FLAG=()
    if [ -n "$QWEN_MODEL" ]; then
        QWEN_FLAG=(--qwen-model "$QWEN_MODEL")
    fi
    
    check_not_running "worker_${worker_id}" "$worker_port"
    nohup env CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" -m serve.model_worker \
        --host "0.0.0.0" \
        --port "$worker_port" \
        --worker-address "$WORKER_ADDR" \
        --controller-address "$CONTROLLER_ADDR" \
        --model-path "$model_path" \
        "${QWEN_FLAG[@]}" \
        $EMOTION_FLAG $QUANTIZATION \
        > "$LOG_DIR/worker_${worker_id}.log" 2>&1 < /dev/null &
    
    echo $! > "$PID_DIR/worker_${worker_id}.pid"
    STARTED+=("worker_${worker_id}")
    echo "Worker $worker_id 已启动 (PID: $!, GPU: $gpu_id)"
    wait_ready "worker_${worker_id}" "http://127.0.0.1:$worker_port/health"
}

# ====== 启动WebUI ======
start_webui() {
    echo "启动 WebUI (端口: $WEBUI_PORT)..."
    
    CONTROLLER_URL="http://localhost:$CONTROLLER_PORT"
    
    start_tts
    check_not_running webui "$WEBUI_PORT"
    local tts_flags=(--tts-api-url "$TTS_API_URL" --tts-mode "$TTS_MODE")
    if [ "$ENABLE_TTS" = false ]; then tts_flags+=(--no-tts); fi
    if [ "$USE_EMOTION" = false ]; then tts_flags+=(--no-emotion); fi
    nohup "$PYTHON" -m serve.web.server \
        --host "$WEBUI_HOST" \
        --port "$WEBUI_PORT" \
        --controller-url "$CONTROLLER_URL" \
        "${tts_flags[@]}" \
        > "$LOG_DIR/webui.log" 2>&1 < /dev/null &
    
    echo $! > "$PID_DIR/webui.pid"
    STARTED+=("webui")
    wait_ready webui "http://127.0.0.1:$WEBUI_PORT/"
    echo "WebUI 已就绪"
}

# ====== 主逻辑 ======
cd "$PROJECT_DIR"

echo "=========================================="
echo "Emomni 分布式服务启动"
echo "=========================================="
echo "项目目录: $PROJECT_DIR"
echo ""

# Select the published checkpoint before checking dependencies.
case "$MODEL_SIZE" in base|small) ;; *) echo "错误: --size 必须为 base 或 small"; exit 1;; esac
if [[ -n "$MODEL_QUANTIZE" && "$MODEL_QUANTIZE" != 4 ]]; then
    echo "错误: --quantize 仅支持 4"; exit 1
fi
if [ "$MODEL_PRESET" = true ] && { [ -n "$MODEL_PATH" ] || [ -n "$MODELS" ] || [ -n "$QUANTIZATION" ]; }; then
    echo "错误: -s/-q 不能与自定义模型路径或 --load-in-* 同时使用"; exit 1
fi
if [ -z "$MODEL_PATH" ] && [ -z "$MODELS" ]; then
    MODEL_PATH="Jotakak/Emomni-v1"
    if [ "$MODEL_SIZE" = small ]; then MODEL_PATH+="-small"; fi
    if [ "$MODEL_QUANTIZE" = 4 ]; then MODEL_PATH+="-bnb-4bit"; fi
fi
if [ "$FOREGROUND" = true ] && { [ "$CONTROLLER_ONLY" = true ] || [ "$WORKER_ONLY" = true ] || [ "$WEBUI_ONLY" = true ]; }; then
    echo "错误: --foreground 仅用于完整服务"; exit 1
fi
if [ "$CONTROLLER_ONLY" = false ] && [ "$WEBUI_ONLY" = false ]; then
    # 单模型验证
    if [ -n "$MODEL_PATH" ] && ! is_valid_model_path "$MODEL_PATH"; then
        echo "错误: 模型路径不存在: $MODEL_PATH"
        echo "提示: 支持本地路径或 HuggingFace Hub 格式 (如 owner/repo-name)"
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
TTS_GPU="${TTS_GPU:-${GPU_ARRAY[0]}}"

# 验证多模型路径
if [ -n "$MODELS" ]; then
    for mp in "${MODEL_ARRAY[@]}"; do
        if ! is_valid_model_path "$mp"; then
            echo "错误: 模型路径不存在: $mp"
            echo "提示: 支持本地路径或 HuggingFace Hub 格式 (如 owner/repo-name)"
            exit 1
        fi
    done
fi

# Validate before installing anything or starting services.
if [ "$WORKER_ONLY" = true ] && [ "$CHECK_DEPS" = false ] && [ -z "$SINGLE_GPU" ]; then
    echo "错误: --worker-only 模式需要指定 --gpu"; exit 1
fi
if [[ ! "${NUM_WORKERS:-1}" =~ ^[1-9][0-9]*$ || ! "$STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误: Worker 数量和 STARTUP_TIMEOUT 必须为正整数"; exit 1
fi
case "$TTS_MODE" in Kokoro|CosyVoice3) ;; *) echo "错误: 无效 TTS 模式"; exit 1;; esac
[[ "$TTS_API_URL" == */v1 ]] || TTS_API_URL="$TTS_API_URL/v1"
services=(controller worker webui)
if [ "$CONTROLLER_ONLY" = true ]; then services=(controller)
elif [ "$WORKER_ONLY" = true ]; then services=(worker)
elif [ "$WEBUI_ONLY" = true ]; then services=(webui)
fi
if [ "$CONTROLLER_ONLY" = false ] && [ "$WORKER_ONLY" = false ] && [ "$ENABLE_TTS" = true ] && [ "$EXTERNAL_TTS" = false ]; then
    if [ "$TTS_MODE" = Kokoro ]; then services+=(kokoro); else services+=(cosyvoice3); fi
fi
check_flags=()
if [ "$CHECK_DEPS" = true ] || [ "$SKIP_INSTALL" = true ]; then check_flags+=(--check); fi
if [ -n "$QUANTIZATION" ]; then check_flags+=(--quantized); fi
"$PYTHON" -m serve.bootstrap --services "${services[@]}" --models "${MODEL_ARRAY[@]}" "${check_flags[@]}"
if [ "$CHECK_DEPS" = true ]; then exit 0; fi

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


# 3. 启动WebUI
start_webui


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
echo "日志目录: $LOG_DIR/"
echo ""
echo "管理命令:"
echo "  停止服务:  bash $0 --stop"
echo "  查看状态:  bash $0 --status"
echo "  查看日志:  bash $0 --logs [controller|worker_0|kokoro|webui|all]"
echo "=========================================="

if [ "$FOREGROUND" = true ]; then
    trap stop_all_services EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    wait -n || true
    echo "服务进程退出，停止其余服务。"
    exit 1
fi
