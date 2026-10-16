#ifndef EMP_AGMPC_FLEXIBLE_INPUT_OUTPUT_H
#define EMP_AGMPC_FLEXIBLE_INPUT_OUTPUT_H
// HARDENED COPY (gc_agmpc baseline), rewritten from upstream flexible_input_output.h (39fd6e2). PATCHES.md P7, P8:
//  * Only the modes the baseline uses are kept: FlexIn  >0 (private input of a party) and -1 (authenticated share,
//    for chaining circuits that share Delta); FlexOut -1 (authenticated share) and q>0 (revealed to party q only).
//    Upstream's -2 (un-authenticated share, no MAC, no consistency) and 0 (public) modes are rejected.
//  * Every opening is MAC-checked; the MACs travel as one SHA-256 digest per receiver (upstream sent a 32-byte
//    struct with 15 uninitialized padding bytes per bit, to every peer, for every bit, whatever its mode).
//  * The masked private inputs and the opened shares are broadcasts: they are recorded in the echo transcript and
//    echo_check() runs before any label is sent, so an input owner cannot give different masked bits to different
//    parties (upstream sent them point-to-point with no check).

using namespace std;

template<int nP>
struct AuthBitShare
{
	bool bit_share;
	block key[nP + 1];
	block mac[nP + 1];
};

template<int nP>
class FlexIn
{
public:
	int len{};
	int party{};

	bool cmpc_associated = false;
	ThreadPool* pool = nullptr;
	bool *value = nullptr;
	block *key[nP + 1];
	block *mac[nP + 1];
	NetIOMP<nP> * io = nullptr;
	block Delta;

	vector<int> party_assignment;     // >0: private input of that party; -1: authenticated share
	vector<bool> plaintext_assignment;
	vector<AuthBitShare<nP>> authenticated_share_assignment;

	FlexIn(int len, int party) {
		this->len = len;
		this->party = party;
		AuthBitShare<nP> empty_abit;
		memset(&empty_abit, 0, sizeof(AuthBitShare<nP>));
		party_assignment.resize(len, 0);
		plaintext_assignment.resize(len, false);
		authenticated_share_assignment.resize(len, empty_abit);
	}

	void associate_cmpc(ThreadPool *associated_pool, bool *associated_value, block *associated_mac[nP + 1], block *associated_key[nP + 1],  NetIOMP<nP> *associated_io, block associated_Delta) {
		this->cmpc_associated = true;
		this->pool = associated_pool;
		this->value = associated_value;
		for(int j = 1; j <= nP; j++) {
			this->mac[j] = associated_mac[j];
			this->key[j] = associated_key[j];
		}
		this->io = associated_io;
		this->Delta = associated_Delta;
	}

	void assign_party(int pos, int which_party) {
		if (which_party != -1 && (which_party < 1 || which_party > nP))
			agmpc_abort(1, "FlexIn: only private (>0) and authenticated-share (-1) inputs are supported");
		party_assignment[pos] = which_party;
	}
	void assign_plaintext_bit(int pos, bool cur_bit) {
		assert(party_assignment[pos] == party);
		plaintext_assignment[pos] = cur_bit;
	}
	void assign_authenticated_bitshare(int pos, AuthBitShare<nP> *abit) {
		assert(party_assignment[pos] == -1);
		memcpy(&authenticated_share_assignment[pos], abit, sizeof(AuthBitShare<nP>));
	}
	int get_length() { return len; }

	void input(bool *masked_input_ret) {
		assert(cmpc_associated);
		vector<vector<int>> owned(nP + 1);
		vector<int> carried;
		for(int i = 0; i < len; i++) {
			if(party_assignment[i] == -1) carried.push_back(i);
			else owned[party_assignment[i]].push_back(i);
		}
		vector<bool> masked(len, false);

		/* (1) open the input masks of party j's wires to j, with MACs (one digest) */
		vector<vector<char>> lam_recv(nP + 1);
		for(int j = 1; j <= nP; ++j) lam_recv[j].resize(owned[party].size());
		vector<future<void>> res;
		vector<future<bool>> chk;
		for(int j = 1; j <= nP; ++j) if(j != party) {
			res.push_back(pool->enqueue([this, &owned, j]() {
				vector<char> b(owned[j].size());
				Hash h;
				for(size_t k = 0; k < owned[j].size(); ++k) { const int i = owned[j][k]; b[k] = value[i]; h.put(&mac[j][i], sizeof(block)); }
				char dg[Hash::DIGEST_SIZE]; h.digest(dg);
				if (!b.empty() && g_tamper.hit("in-mask")) b[0] ^= 1;
				io->send_data(j, b.data(), b.size());
				io->send_data(j, dg, Hash::DIGEST_SIZE);
				io->flush(j);
			}));
			chk.push_back(pool->enqueue([this, &owned, &lam_recv, j]() -> bool {
				io->recv_data(j, lam_recv[j].data(), lam_recv[j].size());
				char dg[Hash::DIGEST_SIZE], dg2[Hash::DIGEST_SIZE];
				io->recv_data(j, dg, Hash::DIGEST_SIZE);
				Hash h;
				for(size_t k = 0; k < owned[party].size(); ++k) {
					block m = key[j][owned[party][k]];
					if (lam_recv[j][k] & 1) m = m ^ Delta;
					h.put(&m, sizeof(block));
				}
				h.digest(dg2);
				return !digest_eq(dg, dg2);
			}));
		}
		joinNclean(res);
		if(joinNcleanCheat(chk)) agmpc_cheat("MAC check of the input masks opened to the input owner failed");

		/* (2) the owner broadcasts x ^ lambda for its wires */
		vector<char> mine(owned[party].size());
		for(size_t k = 0; k < owned[party].size(); ++k) {
			const int i = owned[party][k];
			bool m = plaintext_assignment[i] != value[i];
			for(int j = 1; j <= nP; ++j) if(j != party) m = m != (bool)(lam_recv[j][k] & 1);
			mine[k] = m;
			masked[i] = m;
		}
		vector<vector<char>> theirs(nP + 1);
		for(int j = 1; j <= nP; ++j) if(j != party) theirs[j].resize(owned[j].size());
		for(int j = 1; j <= nP; ++j) if(j != party) {
			res.push_back(pool->enqueue([this, &mine, j]() {
				vector<char> b = mine;
				if (!b.empty() && g_tamper.hit("input-equivocate")) b[0] ^= 1;  // to one peer only
				io->send_data(j, b.data(), b.size());
				io->flush(j);
			}));
			res.push_back(pool->enqueue([this, &theirs, j]() {
				io->recv_data(j, theirs[j].data(), theirs[j].size());
				NetIOMP<nP>::note(j, theirs[j].data(), theirs[j].size());
			}));
		}
		joinNclean(res);
		NetIOMP<nP>::note(party, mine.data(), mine.size());
		for(int j = 1; j <= nP; ++j) if(j != party)
			for(size_t k = 0; k < owned[j].size(); ++k) masked[owned[j][k]] = theirs[j][k] & 1;

		/* (3) carried authenticated shares: open (share ^ input mask) to everyone, MAC-checked */
		const size_t nc = carried.size();
		vector<char> cbit(nc);
		vector<AuthBitShare<nP>> ne(nc);
		for(size_t k = 0; k < nc; ++k) {
			const int i = carried[k];
			const AuthBitShare<nP>& a = authenticated_share_assignment[i];
			ne[k].bit_share = a.bit_share != value[i];
			for(int j = 1; j <= nP; ++j) if(j != party) { ne[k].key[j] = a.key[j] ^ key[j][i]; ne[k].mac[j] = a.mac[j] ^ mac[j][i]; }
			cbit[k] = ne[k].bit_share;
		}
		vector<vector<char>> crecv(nP + 1);
		for(int j = 1; j <= nP; ++j) if(j != party) crecv[j].resize(nc);
		for(int j = 1; j <= nP; ++j) if(j != party) {
			res.push_back(pool->enqueue([this, &cbit, &ne, nc, j]() {
				Hash h;
				for(size_t k = 0; k < nc; ++k) h.put(&ne[k].mac[j], sizeof(block));
				char dg[Hash::DIGEST_SIZE]; h.digest(dg);
				vector<char> b = cbit;
				if (!b.empty() && g_tamper.hit("chain-share")) b[0] ^= 1;
				io->send_data(j, b.data(), b.size());
				io->send_data(j, dg, Hash::DIGEST_SIZE);
				io->flush(j);
			}));
			chk.push_back(pool->enqueue([this, &crecv, &ne, nc, j]() -> bool {
				io->recv_data(j, crecv[j].data(), nc);
				char dg[Hash::DIGEST_SIZE], dg2[Hash::DIGEST_SIZE];
				io->recv_data(j, dg, Hash::DIGEST_SIZE);
				Hash h;
				for(size_t k = 0; k < nc; ++k) { block m = ne[k].key[j]; if (crecv[j][k] & 1) m = m ^ Delta; h.put(&m, sizeof(block)); }
				h.digest(dg2);
				NetIOMP<nP>::note(j, crecv[j].data(), nc);
				return !digest_eq(dg, dg2);
			}));
		}
		joinNclean(res);
		if(joinNcleanCheat(chk)) agmpc_cheat("MAC check of the carried authenticated shares failed");
		NetIOMP<nP>::note(party, cbit.data(), nc);
		for(size_t k = 0; k < nc; ++k) {
			bool m = cbit[k];
			for(int j = 1; j <= nP; ++j) if(j != party) m = m != (bool)(crecv[j][k] & 1);
			masked[carried[k]] = m;
		}

		io->echo_check("FlexIn masked inputs");
		for(int i = 0; i < len; i++) masked_input_ret[i] = masked[i];
	}
};

template<int nP>
class FlexOut
{
public:
	int len{};
	int party{};

	bool cmpc_associated = false;
	ThreadPool* pool = nullptr;
	bool *value = nullptr;
	block *key[nP + 1];
	block *mac[nP + 1];
	block *eval_labels[nP + 1];
	NetIOMP<nP> * io = nullptr;
	block Delta;
	block *labels = nullptr;

	vector<int> party_assignment;     // -1: authenticated share; q > 0: revealed to party q only
	vector<bool> plaintext_results;
	vector<AuthBitShare<nP>> authenticated_share_results;

	FlexOut(int len, int party) {
		this->len = len;
		this->party = party;
		AuthBitShare<nP> empty_abit;
		memset(&empty_abit, 0, sizeof(AuthBitShare<nP>));
		party_assignment.resize(len, -1);
		plaintext_results.resize(len, false);
		authenticated_share_results.resize(len, empty_abit);
	}

	void associate_cmpc(ThreadPool *associated_pool, bool *associated_value, block *associated_mac[nP + 1], block *associated_key[nP + 1], block *associated_eval_labels[nP + 1], block *associated_labels, NetIOMP<nP> *associated_io, block associated_Delta) {
		this->cmpc_associated = true;
		this->pool = associated_pool;
		this->value = associated_value;
		this->labels = associated_labels;
		for (int j = 1; j <= nP; j++) {
			this->mac[j] = associated_mac[j];
			this->key[j] = associated_key[j];
		}
		if (party == ALICE)
			for (int j = 2; j <= nP; j++) this->eval_labels[j] = associated_eval_labels[j];
		this->io = associated_io;
		this->Delta = associated_Delta;
	}

	void assign_party(int pos, int which_party) {
		if (which_party != -1 && (which_party < 1 || which_party > nP))
			agmpc_abort(1, "FlexOut: only authenticated-share (-1) and single-party (>0) outputs are supported");
		party_assignment[pos] = which_party;
	}
	bool get_plaintext_bit(int pos) {
		assert(party_assignment[pos] == party);
		return plaintext_results[pos];
	}
	AuthBitShare<nP> get_authenticated_bitshare(int pos) {
		assert(party_assignment[pos] == -1);
		return authenticated_share_results[pos];
	}
	int get_length() { return len; }

	void output(bool *masked_input_ret, int output_shift) {
		assert(cmpc_associated);
		vector<future<void>> res;
		vector<future<bool>> chk;
		vector<bool> masked_output(len);

		/* (1) the evaluator sends the output labels; each garbler decodes x ^ lambda (and checks the label) */
		if(party == ALICE) {
			for(int i = 0; i < len; i++) masked_output[i] = masked_input_ret[output_shift + i];
			for (int j = 2; j <= nP; j++) {
				res.push_back(pool->enqueue([this, j, output_shift]() {
					vector<block> l(len);
					for(int i = 0; i < len; i++) l[i] = eval_labels[j][output_shift + i];
					if (len > 0 && g_tamper.hit("out-label")) l[0] = l[0] ^ makeBlock(0, 1);
					io->send_data(j, l.data(), sizeof(block) * len);
					io->flush(j);
				}));
			}
			joinNclean(res);
		} else {
			vector<block> l(len);
			io->recv_data(ALICE, l.data(), sizeof(block) * len);
			for(int i = 0; i < len; i++) {
				const block zero_label = labels[i + output_shift];
				const block one_label = zero_label ^ Delta;
				if(cmpBlock(&l[i], &zero_label, 1)) masked_output[i] = false;
				else if(cmpBlock(&l[i], &one_label, 1)) masked_output[i] = true;
				else agmpc_cheat("output label from the evaluator matches neither label");
			}
		}

		/* (2) outputs for party q: every other party opens its lambda share with MACs (one digest) to q */
		vector<vector<int>> to(nP + 1);
		for(int i = 0; i < len; i++) if(party_assignment[i] > 0) to[party_assignment[i]].push_back(i);
		vector<vector<char>> lr(nP + 1);
		for(int j = 1; j <= nP; ++j) if(j != party) lr[j].resize(to[party].size());
		for(int j = 1; j <= nP; ++j) if(j != party) {
			if(!to[j].empty()) res.push_back(pool->enqueue([this, &to, j, output_shift]() {
				vector<char> b(to[j].size());
				Hash h;
				for(size_t k = 0; k < to[j].size(); ++k) { const int w = output_shift + to[j][k]; b[k] = value[w]; h.put(&mac[j][w], sizeof(block)); }
				char dg[Hash::DIGEST_SIZE]; h.digest(dg);
				if (!b.empty() && g_tamper.hit("out-mask")) b[0] ^= 1;
				io->send_data(j, b.data(), b.size());
				io->send_data(j, dg, Hash::DIGEST_SIZE);
				io->flush(j);
			}));
			if(!to[party].empty()) chk.push_back(pool->enqueue([this, &to, &lr, j, output_shift]() -> bool {
				io->recv_data(j, lr[j].data(), lr[j].size());
				char dg[Hash::DIGEST_SIZE], dg2[Hash::DIGEST_SIZE];
				io->recv_data(j, dg, Hash::DIGEST_SIZE);
				Hash h;
				for(size_t k = 0; k < to[party].size(); ++k) { block m = key[j][output_shift + to[party][k]]; if (lr[j][k] & 1) m = m ^ Delta; h.put(&m, sizeof(block)); }
				h.digest(dg2);
				return !digest_eq(dg, dg2);
			}));
		}
		joinNclean(res);
		if(joinNcleanCheat(chk)) agmpc_cheat("MAC check of the output masks failed");
		for(size_t k = 0; k < to[party].size(); ++k) {
			const int i = to[party][k];
			bool v = value[output_shift + i] != masked_output[i];
			for(int j = 1; j <= nP; ++j) if(j != party) v = v != (bool)(lr[j][k] & 1);
			plaintext_results[i] = v;
		}

		/* (3) authenticated-share outputs (upstream logic): P1's share absorbs x ^ lambda, the others re-key P1 */
		for(int i = 0; i < len; i++) {
			if(party_assignment[i] != -1) continue;
			AuthBitShare<nP>& r = authenticated_share_results[i];
			if(party == ALICE) {
				r.bit_share = value[output_shift + i] != masked_output[i];
				for(int j = 1; j <= nP; j++) if(j != party) { r.mac[j] = mac[j][output_shift + i]; r.key[j] = key[j][output_shift + i]; }
			} else {
				r.bit_share = value[output_shift + i];
				for(int j = 1; j <= nP; j++) if(j != party) {
					r.mac[j] = mac[j][output_shift + i];
					r.key[j] = key[j][output_shift + i];
					if(j == ALICE && masked_output[i]) r.key[j] = r.key[j] ^ Delta;
				}
			}
		}
	}
};

#endif //EMP_AGMPC_FLEXIBLE_INPUT_OUTPUT_H
