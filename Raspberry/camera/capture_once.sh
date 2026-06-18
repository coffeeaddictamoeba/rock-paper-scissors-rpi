#!/usr/bin/env bash

set -euo pipefail

output="/dev/shm/brace-rps/latest_frame.bmp"
width=640
height=480
timeout_ms=0
shutter_us=5000
gain=2
awbgains="1.5,1.5"
quality=75
encoding="bmp"
camera_command="rpicam-still"

usage() {
    cat <<'USAGE'
Usage:
  ./capture_once.sh [options]

Options:
  --output PATH       Output image path. Default: /dev/shm/brace-rps/latest_frame.bmp
  --width N           Output image width. Default: 640
  --height N          Output image height. Default: 480
  --timeout-ms N      Camera timeout. Default: 0
  --shutter-us N      Manual shutter time in microseconds. Default: 5000
  --gain N            Manual analogue gain. Default: 2
  --awbgains R,B      Manual AWB gains. Default: 1.5,1.5
  --quality N         Output quality passed to rpicam-still. Default: 75
  --encoding FORMAT   rpicam-still encoding. Default: bmp
  --camera-command C  Camera command. Default: rpicam-still
  --help              Show this help text

Example:
  ./capture_once.sh --output /dev/shm/brace-rps/latest_frame.bmp
USAGE
}

require_value() {
    local option="$1"
    local value="${2:-}"

    if [[ -z "${value}" ]]; then
        echo "Missing value after ${option}" >&2
        exit 1
    fi

    printf '%s' "${value}"
}

is_positive_integer() {
    [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
}

is_non_negative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            output="$(require_value "$1" "${2:-}")"
            shift 2
            ;;
        --width)
            width="$(require_value "$1" "${2:-}")"
            shift 2
            ;;
        --height)
            height="$(require_value "$1" "${2:-}")"
            shift 2
            ;;
        --timeout-ms)
            timeout_ms="$(require_value "$1" "${2:-}")"
            shift 2
            ;;
        --shutter-us)
            shutter_us="$(require_value "$1" "${2:-}")"
            shift 2
            ;;
        --gain)
            gain="$(require_value "$1" "${2:-}")"
            shift 2
            ;;
        --awbgains)
            awbgains="$(require_value "$1" "${2:-}")"
            shift 2
            ;;
        --quality)
            quality="$(require_value "$1" "${2:-}")"
            shift 2
            ;;
        --encoding)
            encoding="$(require_value "$1" "${2:-}")"
            shift 2
            ;;
        --camera-command)
            camera_command="$(require_value "$1" "${2:-}")"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${output}" ]]; then
    echo "--output must not be empty" >&2
    exit 1
fi

if ! is_positive_integer "${width}"; then
    echo "--width must be a positive whole number" >&2
    exit 1
fi

if ! is_positive_integer "${height}"; then
    echo "--height must be a positive whole number" >&2
    exit 1
fi

if ! is_non_negative_integer "${timeout_ms}"; then
    echo "--timeout-ms must be zero or a positive whole number" >&2
    exit 1
fi

if ! is_positive_integer "${shutter_us}"; then
    echo "--shutter-us must be a positive whole number" >&2
    exit 1
fi

if ! is_positive_integer "${quality}" || [[ "${quality}" -gt 100 ]]; then
    echo "--quality must be a whole number between 1 and 100" >&2
    exit 1
fi

if ! command -v "${camera_command}" >/dev/null 2>&1; then
    echo "Camera command not found: ${camera_command}" >&2
    exit 1
fi

output_dir="$(dirname "${output}")"
mkdir -p "${output_dir}"

temp_output="${output_dir}/.${RANDOM}.$(basename "${output}").tmp"
cleanup() {
    rm -f "${temp_output}"
}
trap cleanup EXIT

start_ns="$(date +%s%N)"

"${camera_command}" \
    -n \
    --immediate \
    --timeout "${timeout_ms}" \
    --shutter "${shutter_us}" \
    --gain "${gain}" \
    --awbgains "${awbgains}" \
    --width "${width}" \
    --height "${height}" \
    -q "${quality}" \
    --encoding "${encoding}" \
    --output "${temp_output}"

mv "${temp_output}" "${output}"

end_ns="$(date +%s%N)"
elapsed_ms="$(( (end_ns - start_ns) / 1000000 ))"

echo "Captured ${output}"
echo "Size: ${width}x${height}, encoding: ${encoding}"
echo "Shutter: ${shutter_us} us, gain: ${gain}, AWB gains: ${awbgains}, quality: ${quality}"
echo "Capture command took ${elapsed_ms} ms"
