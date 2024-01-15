import torch.nn
import argparse
from torch.utils.data import DataLoader

import config
from block.Noise import Noise
from utils.datasets import Test_Dataset
from models.encoder_decoder import FED, INL
from utils.utils import *
from utils.metric import *



def decode(noise_name,ratio):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='DECODE')
    parser.add_argument('--noise-type', '-n', default='JPEG', type=str, help='The noise type added to the watermarked images.')
    parser.add_argument('--batch-size', '-b', default=16, type=int, help='The batch size.')
    parser.add_argument('--messages-path', '-m', default="messages", type=str, help='The embedded messages')
    parser.add_argument('--watermarked-image', '-o', default="output_images", type=str, help='The watermarked images')
    parser.add_argument('--test-image', '-ti', default="test_images", type=str, help='The test images')
    parser.add_argument('--testing-model', '-c', default="my", type=str,
                        help='Choose to test the reproduced authors model or my improved model')

    args = parser.parse_args()

    inn_data = Test_Dataset(args.watermarked_image, "png")
    inn_loader = DataLoader(inn_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

    org_data = Test_Dataset(args.test_image, "png")
    org_loader = DataLoader(org_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

    error_history = []

    with torch.no_grad():
        if args.noise_type in ["JPEG", "HEAVY"]:
            # fed_path = os.path.join("experiments",args.noise_type,"FED.pt")
            fed_path = os.path.join("experiments", "reproduction.pt")
            if args.testing_model == 'my':
                fed_path = os.path.join("experiments", "my.pt")
            elif args.testing_model == 'author':
                fed_path = os.path.join("experiments",args.noise_type,"FED.pt")

            fed = FED().to(device)
            load(fed_path, fed, device)
            fed.eval()

            if args.noise_type == "HEAVY":
                inl_path = os.path.join("experiments", args.noise_type, "INL.pt")
                inl = INL().to(device)
                load(inl_path, inl, device)
                inl.eval()

            for idx, (watermarked_images,org_image) in enumerate(zip(inn_loader,org_loader)):
                watermarked_images = watermarked_images.to(device)
                embedded_messgaes = torch.load(os.path.join(args.messages_path,"message_{}.pt".format(idx+1)))

                all_zero = torch.zeros(embedded_messgaes.shape).to(device)

                if args.noise_type == "HEAVY":
                    watermarked_images = inl(watermarked_images.clone(), rev=True)

                if noise_name == 'Cropout':
                    nosice_layer = Noise(['Cropout({r1},{r2})'.format(r1=ratio,r2=ratio)])
                elif noise_name == 'Dropout':
                    nosice_layer = Noise(['Dropout({r1})'.format(r1=ratio)])
                elif noise_name == 'SPNoist':
                    nosice_layer = Noise(['SP({r1})'.format(r1=ratio)])
                elif noise_name == 'JpegComp':
                    nosice_layer = Noise(['JpegTest({r1})'.format(r1=ratio)])
                elif noise_name == 'GaussNoise':
                    nosice_layer = Noise(['GN({r1})'.format(r1=ratio)])
                elif noise_name == 'GaussBlur':
                    nosice_layer = Noise(['GN({r1})'.format(r1=ratio)])
                elif noise_name == 'MedianBlur':
                    nosice_layer = Noise(['MF({r1})'.format(r1=ratio)])

                watermarked_images = nosice_layer((org_image,watermarked_images))
                reversed_img, extracted_messages, _ = fed([watermarked_images, all_zero], rev=True)

                error_rate = decoded_message_error_rate_batch(embedded_messgaes, extracted_messages)

                error_history.append(error_rate)

        else:
            raise ValueError("\"{}\" is not a valid noise type ".format(args.noise_type))

    print('{noisename}, ratio(or weight){ratio}, acc: {acc:.1f}%'.format(noisename=noise_name,ratio=ratio,acc=(1-np.mean(error_history))*100))


if __name__ == '__main__':
    for noise in config.noises:
        decode(noise[0],noise[1])
