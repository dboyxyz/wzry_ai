# Adapted from WZCQ and the local Jetson project; reorganized for this repository.
import argparse
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
import torchvision
from masks import create_masks
from resnet_utils import myResnet, load_policy_backbone
from policy_model import get_model
from config import TransformerConfig, ROOT, WEIGHTS_DIR

@dataclass
class InputState:
    w_down: bool = False
    a_down: bool = False
    s_down: bool = False
    d_down: bool = False
    q_down: bool = False
    attack_down: bool = False
    ai_enabled: bool = True
    running: bool = True
    pending_stop: bool = False
    action_queue: list[str] = field(default_factory=list)
STATE = InputState()
LOCK = threading.Lock()

def get_key_name(key) -> str:
    from pynput import keyboard
    if isinstance(key, keyboard.KeyCode):
        return key.char or ''
    return str(key)

def resolve_direction() -> str:
    if STATE.q_down:
        return 'stop'
    if STATE.w_down and (not STATE.s_down) and (not STATE.a_down) and (not STATE.d_down):
        return 'up'
    if not STATE.w_down and STATE.s_down and (not STATE.a_down) and (not STATE.d_down):
        return 'down'
    if not STATE.w_down and (not STATE.s_down) and STATE.a_down and (not STATE.d_down):
        return 'left'
    if not STATE.w_down and (not STATE.s_down) and (not STATE.a_down) and STATE.d_down:
        return 'right'
    if STATE.w_down and (not STATE.s_down) and STATE.a_down and (not STATE.d_down):
        return 'up-left'
    if STATE.w_down and (not STATE.s_down) and (not STATE.a_down) and STATE.d_down:
        return 'up-right'
    if not STATE.w_down and STATE.s_down and STATE.a_down and (not STATE.d_down):
        return 'down-left'
    if not STATE.w_down and STATE.s_down and (not STATE.a_down) and STATE.d_down:
        return 'down-right'
    return ''

def on_press(key):
    key_name = get_key_name(key)
    with LOCK:
        action = ''
        if key_name == 'w':
            STATE.w_down = True
        elif key_name == 'a':
            STATE.a_down = True
        elif key_name == 's':
            STATE.s_down = True
        elif key_name == 'd':
            STATE.d_down = True
        elif key_name == 'q':
            STATE.q_down = True
        elif key_name == 'i':
            STATE.ai_enabled = not STATE.ai_enabled
        elif key_name == 'Key.space':
            action = 'summoner'
        elif key_name == 'Key.end':
            action = 'last-hit'
        elif key_name == 'Key.page_down':
            action = 'push-tower'
        elif key_name == 'j':
            action = 'skill1'
        elif key_name == 'k':
            action = 'skill2'
        elif key_name == 'l':
            action = 'skill3'
        elif key_name == 'f':
            action = 'recall'
        elif key_name == 'g':
            action = 'heal'
        elif key_name == 'h':
            action = 'summoner'
        elif key_name == 'Key.left':
            action = 'skill1'
        elif key_name == 'Key.down':
            action = 'skill2'
        elif key_name == 'Key.right':
            action = 'skill3'
        elif key_name == 'Key.up':
            STATE.attack_down = True
        if action:
            STATE.action_queue.append(action)

def on_release(key):
    from pynput.keyboard import Key
    key_name = get_key_name(key)
    with LOCK:
        direction_key_released = False
        if key_name == 'w':
            STATE.w_down = False
            direction_key_released = True
        elif key_name == 'a':
            STATE.a_down = False
            direction_key_released = True
        elif key_name == 's':
            STATE.s_down = False
            direction_key_released = True
        elif key_name == 'd':
            STATE.d_down = False
            direction_key_released = True
        elif key_name == 'q':
            STATE.q_down = False
            direction_key_released = True
        elif key_name == 'Key.up':
            STATE.attack_down = False
        if direction_key_released and resolve_direction() == '':
            STATE.pending_stop = True
    if key == Key.esc:
        STATE.running = False
        return False
    return None

def start_listener() -> threading.Thread:
    from pynput.keyboard import Listener
    listener = Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    return listener

def choose_runtime_action(predicted_action: str) -> tuple[str, str, bool]:
    (predicted_move, predicted_skill) = predicted_action.split('_', 1)
    with LOCK:
        direction = resolve_direction()
        manual_action = STATE.action_queue.pop(0) if STATE.action_queue else ''
        attack = STATE.attack_down
        ai_enabled = STATE.ai_enabled
        pending_stop = STATE.pending_stop
        STATE.pending_stop = False
    if pending_stop:
        return ('stop', 'none', ai_enabled)
    if not ai_enabled:
        move = direction or 'keep-move'
        skill = manual_action or ('attack' if attack else 'none')
        return (move, skill, ai_enabled)
    if direction or manual_action or attack:
        move = direction or predicted_move
        if manual_action:
            skill = manual_action
        elif attack:
            skill = 'attack'
        else:
            skill = 'none'
        return (move, skill, ai_enabled)
    return (predicted_move, predicted_skill, ai_enabled)

def load_resnet101(device: torch.device) -> torch.nn.Module:
    backbone = load_policy_backbone()
    backbone.eval()
    backbone.to(device)
    backbone.requires_grad_(False)
    return myResnet(backbone)

def save_frame_and_record(image_dir: Path, frame_index: int, pil_image, record_file, move_action: str, skill_action: str, auto_flag: int) -> None:
    image_path = image_dir / f'{frame_index}.jpg'
    pil_image.save(image_path)
    record = {'frame_id': str(frame_index), 'movement': move_action, 'skill': skill_action, 'control_mode': auto_flag}
    json.dump(record, record_file, ensure_ascii=False)
    record_file.write('\n')

def main() -> int:
    parser = argparse.ArgumentParser(description='Capture -> features -> policy -> Android touch control')
    parser.add_argument('--serial', default=None, help='ADB device serial')
    parser.add_argument('--display', default=os.environ.get('DISPLAY', ':1'), help='X11 display, for example :1')
    parser.add_argument('--top', type=int, default=65)
    parser.add_argument('--left', type=int, default=60)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=432)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--weights-dir', type=Path, default=WEIGHTS_DIR)
    parser.add_argument('--model-name', default='policy.pt')
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--loop-delay', type=float, default=0.22, help='Minimum decision cycle in seconds')
    parser.add_argument('--save-data', action='store_true', help='Save frames and executed action records')
    parser.add_argument('--data-dir', default='./data/runs')
    args = parser.parse_args()
    from linux_capture import CaptureConfig, MSSCapture
    from linux_control import ScrcpyController
    os.environ['DISPLAY'] = args.display
    os.environ['WZRY_CAPTURE_TOP'] = str(args.top)
    os.environ['WZRY_CAPTURE_LEFT'] = str(args.left)
    os.environ['WZRY_CAPTURE_WIDTH'] = str(args.width)
    os.environ['WZRY_CAPTURE_HEIGHT'] = str(args.height)
    weights_dir = Path(args.weights_dir).resolve()
    model_path = weights_dir / args.model_name
    print(f'using_model_name={args.model_name}')
    print(f'using_weights_dir={weights_dir}')
    print(f'using_model_path={model_path}')
    print(f'model_exists={model_path.is_file()}')
    if not model_path.is_file():
        raise SystemExit(f'Policy checkpoint not found: {model_path}')
    runtime_device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    resnet101 = load_resnet101(runtime_device)
    cfg = TransformerConfig(load_weights=str(weights_dir))
    model = get_model(cfg, 130, args.model_name).to(runtime_device).eval()
    model.requires_grad_(False)
    with (ROOT / 'config/id_to_action.json').open(encoding='utf-8') as f:
        number_to_word = json.load(f)
    with (ROOT / 'config/action_to_id.json').open(encoding='utf-8') as f:
        word_to_number = json.load(f)
    with (ROOT / 'config/touch_scripts.json').open(encoding='utf-8') as f:
        action_to_script = json.load(f)
    controller = ScrcpyController(serial=args.serial)
    listener = start_listener()
    feature_seq = None
    op_seq = [128]
    move_command = 'stop'
    step = 0
    data_dir = Path(args.data_dir)
    record_file = None
    image_dir = None
    if args.save_data:
        data_dir.mkdir(parents=True, exist_ok=True)
        image_dir = data_dir / str(int(time.time()))
        image_dir.mkdir(parents=True, exist_ok=True)
        record_file = open(data_dir / f'{image_dir.name}_actions.jsonl', 'w', encoding='utf-8')
    try:
        with MSSCapture(CaptureConfig.from_env()) as capture, torch.no_grad():
            while STATE.running:
                step_start = time.perf_counter()
                pil_frame = capture.grab_pil()
                frame_np = np.asarray(pil_frame)
                img_t = torch.tensor(frame_np, device=runtime_device, dtype=torch.float32).unsqueeze(0)
                img_t = img_t.permute(0, 3, 2, 1) / 255.0
                (_, feat) = resnet101(img_t)
                feat = feat.reshape(1, 6 * 6 * 2048)
                if feature_seq is None:
                    feature_seq = feat
                elif feature_seq.size(0) < 300:
                    feature_seq = torch.cat((feature_seq, feat), 0)
                else:
                    feature_seq = torch.cat((feature_seq[1:300, :], feat), 0)
                seq_t = torch.tensor(op_seq, device=runtime_device, dtype=torch.long).unsqueeze(0)
                (_, trg_mask) = create_masks(seq_t, seq_t, runtime_device)
                (logits, value) = model(feature_seq.unsqueeze(0), seq_t, trg_mask)
                prob = torch.softmax(logits[:, -1, :], dim=-1)
                action = int(torch.distributions.Categorical(prob).sample().item())
                action_word = number_to_word[str(action)]
                (move_action, skill_action, ai_enabled) = choose_runtime_action(action_word)
                if move_action != 'keep-move' and move_action != move_command:
                    move_command = move_action
                    controller.send_legacy_script(action_to_script[move_action])
                if skill_action != 'none' and skill_action in action_to_script:
                    controller.send_legacy_script(action_to_script[skill_action])
                combined_action = f'{move_action}_{skill_action}'
                history_move = move_command if move_action == 'keep-move' else move_action
                executed_action_word = f'{history_move}_{skill_action}'
                if executed_action_word not in word_to_number:
                    raise RuntimeError(f'Executed action is missing from the vocabulary: {executed_action_word}')
                executed_action = int(word_to_number[executed_action_word])
                op_seq.append(executed_action)
                if len(op_seq) > 300:
                    op_seq = op_seq[-300:]
                if args.save_data and image_dir is not None and (record_file is not None):
                    save_frame_and_record(image_dir=image_dir, frame_index=step, pil_image=pil_frame, record_file=record_file, move_action=history_move, skill_action=skill_action, auto_flag=int(not ai_enabled))
                if args.print_every > 0 and step % args.print_every == 0:
                    print(f'step={step} predicted={action_word} runtime={combined_action} history={executed_action_word} p={prob[0, action].item():.4f} v={value[:, -1, :].item():.4f} ai={int(ai_enabled)}')
                elapsed = time.perf_counter() - step_start
                delay = args.loop_delay - elapsed
                if delay > 0:
                    time.sleep(delay)
                step += 1
    finally:
        listener.stop()
        controller.stop()
        if record_file is not None:
            record_file.close()
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
