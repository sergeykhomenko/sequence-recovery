//
// Created by Serhii Khomenko on 12.10.2024.
//

#include <fstream>
#include <iostream>
#include <random>
#include <cstring>

using namespace std;

const int MIN_SEQ_LENGTH = 32;
// const int MAX_SEQ_LENGTH = 1024;
const int MAX_SEQ_LENGTH = 128;
const int SAMPLES_PER_CHALLENGE = 5;
const int SAMPLES_PER_LENGTH = 20;

const int NUMERIC_SMALL_LENGTH = 16;
const int NUMERIC_LARGE_LENGTH = 128;

char** randomly_override_seq(
    const char* seq,
    mt19937 engine,
    uniform_int_distribution<> bad_sector_dist,
    uniform_int_distribution<> sector_loc_dist,
    uniform_int_distribution<> seq_replacements_dist
) {

    const unsigned int seq_length = strlen(seq);
    char ** res = new char* [SAMPLES_PER_CHALLENGE];
    // todo: book already taken slots, to make the overrides more spreaded over the sequence
    for (int i = 0; i < SAMPLES_PER_CHALLENGE; i++) {
        res[i] = new char[seq_length];
        strcpy(res[i], seq);

        int sectors_lim = bad_sector_dist(engine);
        while (sectors_lim > 0) {
            int s_size = seq_replacements_dist(engine);
            int first_occ = sector_loc_dist(engine);

            if (s_size > sectors_lim) {
                s_size = sectors_lim;
            }

            if (first_occ + s_size > seq_length) {
                s_size = static_cast<int>(seq_length) - first_occ - 1;
            }

            for (int j = 0; j < s_size; j++) {
                res[i][first_occ + j] = '0';
            }

            sectors_lim -= s_size;
        }
    }

    return res;
}

void print_challenges(const char* name, char** list, int size) {
    for (int i = 0; i < size; i++) {
        if (i == 0) {
            cout << name << ":\t";
        } else {
            cout << "\t";
        }

        cout << list[i] << endl;
    }
}

void append_dataset(ofstream &dataset, char* original, char** generated, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        dataset << "{\"messages\": [";
        dataset << "{\"role\": \"system\", \"content\": \"You are a recovery chatbot with the goal of reconstructing corrupted sequences using a user-provided alphabet. As input, you will receive an alphabet and a sequence that needs to be recovered. Your task is to restore the sequence. Any missing elements in the sequence are replaced by 0. The recovered elements should match the length of the missing parts (0s) in the sequence exactly.\"},";
        dataset << "{\"role\": \"user\", \"content\": \"Alphabet: [A, B, C, D]\\nSequence to recover: ";
        dataset << generated[i];
        dataset << "\"}, {\"role\": \"assistant\", \"content\": \"";
        dataset << original;
        dataset << "\"}]}";
        dataset << endl;
    }
}

int main() {
    // Alphabets
    const char *small = "ABCD";

    random_device rd;
    mt19937 engine(rd());
    uniform_int_distribution<> seq_length_dist(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH);
    ofstream dataset("dataset.jsonl");

    unsigned long small_size = strlen(small);
    uniform_int_distribution<> small_alphabet_dist(0, static_cast<int>(small_size) - 1);

    int seq_length = seq_length_dist(engine);
    char* seq = new char[seq_length];
    for (int i = 0; i < seq_length; i++) {
        seq[i] = small[small_alphabet_dist(engine)];
    }

    cout << "Initial sequence: " << endl;
    cout << seq << endl;
    cout << "Challenges:" << endl;

    uniform_int_distribution<> seq_sector_loc_dist(0, seq_length - 1);

    // - small: 1-2 missing elements, not more than 10% of the sequence
    char ** easy_challenges = randomly_override_seq(
        seq,
        engine,
        uniform_int_distribution<> (1, 2),
        uniform_int_distribution<> (1, seq_length / 10),
        seq_sector_loc_dist
    );
    print_challenges("Easy", easy_challenges, SAMPLES_PER_CHALLENGE);
    append_dataset(dataset, seq, easy_challenges, SAMPLES_PER_CHALLENGE);

    // - medium: not more than 4 missing elements, up to 15% of the sequence
    char** medium_challenges = randomly_override_seq(
        seq,
        engine,
        uniform_int_distribution<> (1, 4),
        uniform_int_distribution<> (1, (seq_length * 3) / 20),
        seq_sector_loc_dist
    );
    print_challenges("Medium", medium_challenges, SAMPLES_PER_CHALLENGE);
    append_dataset(dataset, seq, easy_challenges, SAMPLES_PER_CHALLENGE);


    // - hard: gaps size 4-10, 20-30% missing
    char** hard_challenges = randomly_override_seq(
        seq,
        engine,
        uniform_int_distribution<> (4, 10),
        uniform_int_distribution<> (seq_length / 5, (seq_length * 3) / 10),
        seq_sector_loc_dist
    );
    print_challenges("Hard", hard_challenges, SAMPLES_PER_CHALLENGE);
    append_dataset(dataset, seq, easy_challenges, SAMPLES_PER_CHALLENGE);

    // - ultra-hard: minimal gap: 8 elements, 30-50% of sequence is missing
    char** ultra_challenges = randomly_override_seq(
        seq,
        engine,
        uniform_int_distribution<> (4, 10),
        uniform_int_distribution<> (seq_length / 5, (seq_length * 3) / 10),
        seq_sector_loc_dist
    );
    print_challenges("Ultra", ultra_challenges, SAMPLES_PER_CHALLENGE);
    append_dataset(dataset, seq, easy_challenges, SAMPLES_PER_CHALLENGE);


    dataset.close();

    return 0;
}
