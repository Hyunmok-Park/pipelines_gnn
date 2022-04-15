import argparse

import bentoml

def main(bento_file):
    print("===============")
    print("IMPORTING BENTO: {}".format(bento_file))
    print("===============")
    bentoml.import_bento(bento_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bento_file')
    args = parser.parse_args()

    main(args.bento_file)
