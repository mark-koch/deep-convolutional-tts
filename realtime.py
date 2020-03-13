import argparse
import simpleaudio
from networks import SSRN, Text2Mel
from config import Config
from text_processing import *
from audio_processing import spectrogram2wav
from data import *


parser = argparse.ArgumentParser()
parser.add_argument('--t2m', dest='text2mel_path', required=True, help='Path to Text2Mel save file')
parser.add_argument('--ssrn', dest='ssrn_path', required=True, help='Path to SSRN save file')
parser.add_argument('--lang', dest='language', default="en", required=False, help='Language of the text')


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert os.path.exists(args.text2mel_path), "File '{}' does not exist!".format(args.text2mel_path)
    assert os.path.exists(args.ssrn_path), "File '{}' does not exist!".format(args.text2mel_path)

    # Restore config
    state_t2m = torch.load(args.text2mel_path, map_location=device)
    config_t2m = state_t2m["config"]
    state_ssrn = torch.load(args.ssrn_path, map_location=device)
    config_ssrn = state_ssrn["config"]
    if config_ssrn != config_t2m:
        print("WARNING: Text2Mel and SSRN have different saved configs. Will use Text2Mel config!")
    Config.set_config(config_t2m)

    # Load networks
    print("Loading Text2Mel...")
    text2mel = Text2Mel().to(device)
    text2mel.eval()
    text2mel_step = state_t2m["global_step"]
    text2mel.load_state_dict(state_t2m["model"])

    print("Loading SSRN...")
    ssrn = SSRN().to(device)
    ssrn.eval()
    ssrn_step = state_ssrn["global_step"]
    ssrn.load_state_dict(state_ssrn["model"])

    while True:
        text = input("> ")
        text = spell_out_numbers(text, args.language)
        text = normalize(text)
        text = text + Config.vocab_end_of_text
        text = vocab_lookup(text)

        L = torch.tensor(text, device=device, requires_grad=False).unsqueeze(0)
        S = torch.zeros(1, Config.max_T, Config.F, requires_grad=False, device=device)
        previous_position = torch.zeros(1, requires_grad=False, dtype=torch.long, device=device)
        previous_att = torch.zeros(1, len(text), Config.max_T, requires_grad=False, device=device)
        for t in range(Config.max_T - 1):
            _, Y, A, current_position = text2mel.forward(L, S,
                                                         force_incremental_att=False,
                                                         previous_att_position=previous_position,
                                                         previous_att=previous_att,
                                                         current_time=t)
            S[:, t + 1, :] = Y[:, t, :].detach()
            previous_position = current_position.detach()
            previous_att = A.detach()

        # Generate linear spectrogram.
        _, Z = ssrn.forward(S.transpose(1, 2))
        Z = Z.transpose(1, 2).detach().cpu().numpy()
        wav = spectrogram2wav(Z[0])
        wav = np.concatenate([np.zeros(10000), wav], axis=0)  # Silence at the beginning
        wav *= 32767 / max(abs(wav))
        wav = wav.astype(np.int16)

        po = simpleaudio.play_buffer(wav, num_channels=1, bytes_per_sample=2, sample_rate=Config.sample_rate)
        po.wait_done()


