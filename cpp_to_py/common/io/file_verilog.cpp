#include "common/db/Database.h"

using namespace db;

bool isVerilogSymbol(unsigned char c) {
    static char symbols[256] = {0};
    static bool inited = false;
    if (!inited) {
        symbols[(int)'('] = 1;
        symbols[(int)')'] = 1;
        symbols[(int)','] = 1;
        symbols[(int)'.'] = 1;
        symbols[(int)':'] = 1;
        symbols[(int)';'] = 1;
        // symbols[(int)'/'] = 1;
        symbols[(int)'#'] = 1;
        symbols[(int)'['] = 1;
        symbols[(int)']'] = 1;
        symbols[(int)'{'] = 1;
        symbols[(int)'}'] = 1;
        symbols[(int)'*'] = 1;
        symbols[(int)'\"'] = 1;
        symbols[(int)'\\'] = 1;

        symbols[(int)' '] = 2;
        symbols[(int)'\t'] = 2;
        symbols[(int)'\n'] = 2;
        symbols[(int)'\r'] = 2;
        inited = true;
    }
    return symbols[(int)c] != 0;
}
bool readVerilogLine(istream &is, vector<string> &tokens) {
    tokens.clear();
    string line;
    while (is && tokens.empty()) {
        // read next line in
        getline(is, line);

        char token[1024] = {0};
        int lineLen = (int)line.size();
        int tokenLen = 0;
        for (int i = 0; i < lineLen; i++) {
            char c = line[i];
            if (isVerilogSymbol(c)) {
                if (tokenLen > 0) {
                    token[tokenLen] = (char)0;
                    tokens.push_back(string(token));
                    token[0] = (char)0;
                    tokenLen = 0;
                }
                if (c == ';') {
                    tokens.push_back(";");
                }
            } else {
                token[tokenLen++] = c;
                if (tokenLen > 1024) {
                    // TODO: unhandled error
                    tokens.clear();
                    return false;
                }
            }
        }
        // line finished, something else in token
        if (tokenLen > 0) {
            token[tokenLen] = (char)0;
            tokens.push_back(string(token));
            tokenLen = 0;
        }
    }
    return !tokens.empty();
}

bool Database::readVerilog(const std::string &file) {
    ifstream fs(file.c_str());
    if (!fs.good()) {
        logger.error("cannot open verilog file: %s", file.c_str());
        return false;
    }

    vector<string> tokens;
    const int StatusNone = 0;
    const int StatusModule = 1;
    const int StatusInput = 2;
    const int StatusOutput = 3;
    const int StatusWire = 4;
    const int StatusGate = 5;

    int status = StatusNone;
    bool finished = true;
    // loop every line in
    while (readVerilogLine(fs, tokens)) {
        if (tokens[0] == "//") {
            continue;
        }
        if (finished) {
            if (tokens[0] == "endmodule") {
                status = StatusNone;
                tokens.erase(tokens.begin());
                finished = true;
            } else if (tokens[0] == "module") {
                status = StatusModule;
                tokens.erase(tokens.begin());
                finished = false;
            } else if (tokens[0] == "input") {
                status = StatusInput;
                tokens.erase(tokens.begin());
                finished = false;
            } else if (tokens[0] == "output") {
                status = StatusOutput;
                tokens.erase(tokens.begin());
                finished = false;
            } else if (tokens[0] == "wire") {
                status = StatusWire;
                tokens.erase(tokens.begin());
                finished = false;
            } else {
                status = StatusGate;
                finished = false;
            }
        }

        if (tokens.back() == ";") {
            tokens.pop_back();
            finished = true;
        }

        if (status == StatusModule) {
            // ignore
        } else if (status == StatusInput || status == StatusOutput) {
            for (unsigned i = 0; i != tokens.size(); ++i) {
                string pinName(tokens[i]);
                IOPin *iopin = getIOPin(pinName);
                if (!iopin) {
                    logger.error("io pin not found: %s", pinName.c_str());
                }
                Net *net = addNet(pinName);
                net->addPin(iopin->pin);
            }
        } else if (status == StatusWire) {
            for (int i = 0; i < (int)tokens.size(); i++) {
                string netName(tokens[i]);
                this->addNet(netName);
            }
        } else if (status == StatusGate) {
            string cellName(tokens[1]);
            Cell *cell = this->getCell(cellName);
            int nPins = (tokens.size() - 2) / 2;
            for (int i = 0; i < nPins; i++) {
                string pinName(tokens[2 + i * 2]);
                string netName(tokens[3 + i * 2]);
                Pin *pin = cell->pin(pinName);
                Net *net = this->getNet(netName);
                net->addPin(pin);
            }
        }
    }

    fs.close();
    return true;
}
