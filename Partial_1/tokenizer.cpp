#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>

using namespace std;
using namespace std::chrono;

class Tokenizer {
private:
    string delimiter;

public:
    Tokenizer() {
        delimiter = " \t\n\r\f\v!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    }
    
    string verify_word(const string& text) {
        string numbers = "0123456789";
        bool is_only_number = true;
        string word = "";
        

        for (char c : text) {
            if (numbers.find(c) == string::npos) {
                is_only_number = false;
                break;
            }
        }
        
        if (is_only_number) {
            word = text;
        } else {
            
            for (char c : text) {
                if (numbers.find(c) == string::npos) {
                    word += c;
                }
            }
        }
        
        return word;
    }

    vector<string> tokenize(const string& text) {
    
        auto start = high_resolution_clock::now();
        
        vector<string> tokens;
        int n = text.length();
        
        int i = 0;
        int j = 0;
        
        while (i <= n - 1) {
            if ((delimiter.find(text[i]) != string::npos) && 
                (delimiter.find(text[j]) != string::npos)) {
                j++;
            } else if (delimiter.find(text[i]) != string::npos) {
                if (i > j) {  
                    string word_verified = verify_word(text.substr(j, i - j));
                    if (!word_verified.empty()) {  
                        tokens.push_back(word_verified);
                    }
                }
                j = i + 1;
            }
            i++;
        }
        
        if (j < n) {
            string word_verified = verify_word(text.substr(j));
            if (!word_verified.empty()) {
                tokens.push_back(word_verified);
            }
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        
        cout << "Time: " << duration.count() << " microseconds" << endl;

        return tokens;
    }
};

int main() {
    Tokenizer tokenizer;
    string text = " Hoy hay clase123 de PNL. Hay jun23ta a las 1945. o   holavcghv.   gcv  Tienen tarea á ñ ";
    
    cout << "Tokens: ";
    
    vector<string> tokens = tokenizer.tokenize(text);
    
    cout << "[";
    for (size_t i = 0; i < tokens.size(); i++) {
        cout << "\"" << tokens[i] << "\"";
        if (i < tokens.size() - 1) {
            cout << ", ";
        }
    }
    cout << "]" << endl;
    
    return 0;
}