//
// Created by Serhii Khomenko on 12.10.2024.
//

#include <iostream>
#include <random>
#include <cstring>

using namespace std;

const int MIN_SEQ_LENGTH = 32;
// const int MAX_SEQ_LENGTH = 1024;
const int MAX_SEQ_LENGTH = 128;
const int SAMPLES_PER_CHALLENGE = 5;

const int NUMERIC_SMALL_LENGTH = 16;
const int NUMERIC_LARGE_LENGTH = 128;

char** randomly_override_seq(char* seq, mt19937 engine, int max_bad_sect, int max_seq_repl, uniform_int_distribution<> sector_loc_dist) {
    uniform_int_distribution<> bad_sector_dist(1, max_bad_sect);
    uniform_int_distribution<> seq_replacements_dist(1, max_seq_repl);

    const unsigned int seq_length = strlen(seq);
    char ** res = new char* [SAMPLES_PER_CHALLENGE];
    // todo: book already taken slots, to make the overrides more spreaded over the sequence
    for (int i = 0; i < SAMPLES_PER_CHALLENGE; i++) {
        res[i] = new char[seq_length];
        strcpy(res[i], seq);

        int sectors_lim = seq_replacements_dist(engine);
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
                res[i][first_occ + j] = *"0";
            }

            sectors_lim -= s_size;
        }
    }

    return res;
}

int main() {
    // Alphabets
    const char *small = "ABCD";

    random_device rd;
    mt19937 engine(rd());
    uniform_int_distribution<> seq_length_dist(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH);

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
    char ** easy_challenges = randomly_override_seq(seq, engine, 2, seq_length / 10, seq_sector_loc_dist);
    for (int i = 0; i < SAMPLES_PER_CHALLENGE; i++) {
        cout << (i == 0 ? "Easy:\t\t" : "\t\t\t") << easy_challenges[i] << endl;
    }

    // - medium: not more than 4 missing elements, up to 15% of the sequence
    char** medium_challenges = randomly_override_seq(seq, engine, 4, (seq_length * 3) / 20, seq_sector_loc_dist);
    for (int i = 0; i < SAMPLES_PER_CHALLENGE; i++) {
        cout << (i == 0 ? "Medium:\t\t" : "\t\t\t") << medium_challenges[i] << endl;
    }


    // - large: gaps size 4-10, 20-30% missing



    // - extra large: minimal gap: 8 elements, 30-50% of sequence is missing


    cout << "Large:\t" << endl;
    cout << "X-Large:\t" << endl;

    return 0;
}
