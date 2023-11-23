from tqdm import tqdm
import argparse

vocab = ['<pad>', '<s>', '</s>', '<unk>', 'ই', '3', 'হ', '…', 'ল', '্', 'ৈ', 'ো', '৪', 'ধ', 'উ', 'া', 'ঞ', 'F', 'অ', 'ও', 'ট', 'খ', 'ড়', 'স', '০', 'ম', 'ং', 'ৌ', 'গ', 'ঃ', '\u200c', 'থ', 'e', 'ি', 'ষ', '৯', '়', 'চ', 'শ', 'ৗ', 'ঊ', '৬', 'ঈ', 'ঋ', 'ঠ', 'ত', 'এ', '৫', 'আ', 'ছ', 'ূ', 'ব', 'ঐ', 'প', 'ী', 'ড', '৭', 'ণ', 'ফ', 'ু', 'ৃ', '১', '|', '৮', '\u200d', 'i', 'ৰ', 'ঔ', 'ভ', '\u200e', 'ঙ', 'ৎ', 'ঘ', 'দ', '২', 'ঝ', 'l', 'য়', 'জ', 'ক', 'ন', 'য', 'ে', 'র', '৩', 'ঢ', 'ঁ']

def make_lexicon_lst_from_txt_file(txt_file, path_to_save_lexicon):
    with open(txt_file) as f:
        content_list = f.readlines()

    content_list = [i.replace("\n", "") for i in content_list]

    with open(path_to_save_lexicon, "w+") as f:
        for line in tqdm(content_list):
            print(str(line + "\t" + " ".join(list(line.replace("/n", "").replace(" ", "|").strip())) + " |"), file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run')
    parser.add_argument('--txt-file', type=str)
    parser.add_argument('--path-to-save-lexicon', type=str)
    args_local = parser.parse_args()
    txt_file = args_local.txt_file
    path_to_save_lexicon = args_local.path_to_save_lexicon
    make_lexicon_lst_from_txt_file(txt_file, path_to_save_lexicon)
