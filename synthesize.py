import argparse
import io
import multiprocessing
from networks import SSRN, Text2Mel
from config import Config
from text_processing import *
from audio_processing import spectrogram2wav
from data import *
from scipy.io.wavfile import write


parser = argparse.ArgumentParser()
parser.add_argument('--t2m', dest='text2mel_path', required=True, help='Path to Text2Mel save file')
parser.add_argument('--ssrn', dest='ssrn_path', required=True, help='Path to SSRN save file')
parser.add_argument('--lang', dest='language', default="en", required=False, help='Language of the text')
parser.add_argument('-o', dest='output_path', required=False, default="output.wav",
                    help='Destination path for wav file')
parser.add_argument('--workers', dest='num_workers', type=int, default=None, required=False,
                    help='Number of processes to use for final wav generation')
parser.add_argument('--t2m_bs', dest='text2mel_batch_size', type=int, default=32, required=False,
                    help='Batch size to use for the Text2Mel network. This can typically be higher than the batch size '
                         'for SSRN')
parser.add_argument('--ssrn_bs', dest='ssrn_batch_size', type=int, default=4, required=False,
                    help='Batch size to use for the SSRN network')
parser.add_argument('-n', '--max_N', dest='max_N', type=int, default=180, required=False,
                    help='Maximum number of characters per text chunk.')
parser.add_argument('-t', '--max_T', dest='max_T', type=int, default=210, required=False,
                    help='Maximum number of mel frames per generated audio chunk.')
parser.add_argument('text_path', help='Path to the text file to be synthesized')


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert os.path.exists(args.text2mel_path), "File '{}' does not exist!".format(args.text2mel_path)
    assert os.path.exists(args.ssrn_path), "File '{}' does not exist!".format(args.text2mel_path)
    assert os.path.exists(args.text_path), "File '{}' does not exist!".format(args.text_path)

    # Restore config
    state_t2m = torch.load(args.text2mel_path, map_location=device)
    config_t2m = state_t2m["config"]
    state_ssrn = torch.load(args.ssrn_path, map_location=device)
    config_ssrn = state_ssrn["config"]
    if config_ssrn != config_t2m:
        print("WARNING: Text2Mel and SSRN have different saved configs. Will use Text2Mel config!")
    Config.set_config(config_t2m)

    # Read input file
    with io.open(args.text_path, "r", encoding="utf-8") as file:
        text = file.read()

    text = remove_abbreviations(text, args.language)
    texts = split_text(text, max_len=args.max_N-1)  # -1 because we need to add an EOT at the end
    # Split the text into batches
    batches = []
    for i in range(0, len(texts), args.text2mel_batch_size):
        batches.append(texts[i:i + args.text2mel_batch_size])

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

    # Setup multiprocessing for wav generation
    pool = multiprocessing.Pool(processes=args.num_workers)
    wavs = []

    for batch in batches:
        # Process the texts
        for i in range(len(batch)):
            batch[i] = normalize(batch[i])
            batch[i] = batch[i] + Config.vocab_end_of_text
            batch[i] = vocab_lookup(batch[i])

        # Create text tensor L with zero padding
        max_text_len = max(len(text) for text in batch)
        L = torch.zeros(len(batch), max_text_len, dtype=torch.long, device=device, requires_grad=False)
        for i in range(len(batch)):
            L[i, :len(batch[i])] = torch.tensor(batch[i], dtype=torch.long, device=device)

        S = torch.zeros(len(batch), args.max_T, Config.F, requires_grad=False, device=device)
        previous_position = torch.zeros(len(batch), requires_grad=False, dtype=torch.long, device=device)
        previous_att = None  # torch.zeros(len(batch), max_text_len, Config.max_T, requires_grad=False, device=device)
        for t in range(args.max_T-1):
            print(t)
            _, Y, A, current_position = text2mel.forward(L, S,
                                                         force_incremental_att=True,
                                                         previous_att_position=previous_position,
                                                         previous_att=previous_att,
                                                         current_time=t)
            S[:, t+1, :] = Y[:, t, :].detach()
            previous_position = current_position.detach()
            previous_att = A.detach()

        # Generate linear spectrogram. We need to rebatch
        for i in range(0, S.shape[0], args.ssrn_batch_size):
            Y = S[i:i + args.ssrn_batch_size]
            _, Z = ssrn.forward(Y.transpose(1, 2))
            Z = Z.transpose(1, 2).detach().cpu().numpy()

            for j in range(Z.shape[0]):
                wavs.append(pool.apply_async(spectrogram2wav, (Z[j], )))

    # Wait for the workers to finish
    for i in range(len(wavs)):
        wavs[i] = wavs[i].get()

    final_wav = np.concatenate(wavs)
    write(args.output_path, Config.sample_rate, final_wav)

    print("Finished!")


