#ifndef CMPC_H__
#define CMPC_H__
#include "fpremp.h"
#include "abitmp.h"
#include "netmp.h"
#include "flexible_input_output.h"
#include <emp-tool/emp-tool.h>
using namespace emp;

template<int nP>
class CMPC { public:
	const static int SSP = 5;//5*8 in fact...
	const block MASK = makeBlock(0x0ULL, 0xFFFFFULL);
	FpreMP<nP>* fpre = nullptr;
	block* mac[nP+1];
	block* key[nP+1];
	bool* value;

	block * preprocess_mac[nP+1];
	block * preprocess_key[nP+1];
	bool* preprocess_value;

	block * sigma_mac[nP+1];
	block * sigma_key[nP+1];
	bool * sigma_value;

	block * ANDS_mac[nP+1];
	block * ANDS_key[nP+1];
	bool * ANDS_value;

	block * labels;
	BristolFormat * cf;
	NetIOMP<nP> * io;
	int num_ands = 0, num_in;
	int party, total_pre, ssp;
	ThreadPool * pool;
	block Delta;
		
	block (*GTM)[4][nP+1];
	block (*GTK)[4][nP+1];
	bool (*GTv)[4];
	block (*GT)[nP+1][4][nP+1];
	block * eval_labels[nP+1];
	PRP prp;
	bool owns_fpre = true;
	uint64_t inst = 0;          // PATCH P1: per-instance hash tweak (set after function_independent)
	block Delta2, Delta4;       // 2*Delta, 4*Delta in GF(2^128)
	// PATCH P9: one FpreMP (base OTs, COT extenders, Delta) shared by every instance of the run.
	CMPC(NetIOMP<nP> * io[2], ThreadPool * pool, int party, BristolFormat * cf, FpreMP<nP>* shared_fpre, int ssp) {
		owns_fpre = false; fpre = shared_fpre; init(io, pool, party, cf, ssp);
	}
	CMPC(NetIOMP<nP> * io[2], ThreadPool * pool, int party, BristolFormat * cf, bool * _delta = nullptr, int ssp = 40) {
		fpre = new FpreMP<nP>(io, pool, party, _delta, ssp); init(io, pool, party, cf, ssp);
	}
	void init(NetIOMP<nP> * io[2], ThreadPool * pool, int party, BristolFormat * cf, int ssp) {
		this->party = party;
		this->io = io[0];
		this->cf = cf;
		this->ssp = ssp;
		this->pool = pool;

		for(int i = 0; i < cf->num_gate; ++i) {
			if (cf->gates[4*i+3] == AND_GATE)
				++num_ands;
		}
		num_in = cf->n1+cf->n2;
		total_pre = num_in + num_ands + 3*ssp;
		Delta = fpre->Delta;
		Delta2 = gf128_double(Delta);
		Delta4 = gf128_double(Delta2);

		if(party == 1) {
			GTM = new block[num_ands][4][nP+1];
			GTK = new block[num_ands][4][nP+1];
			GTv = new bool[num_ands][4];
			GT = new block[num_ands][nP+1][4][nP+1];
		}

		labels = new block[cf->num_wire];
		for(int i  = 1; i <= nP; ++i) {
			key[i] = new block[cf->num_wire];
			mac[i] = new block[cf->num_wire];
			ANDS_key[i] = new block[num_ands*3];
			ANDS_mac[i] = new block[num_ands*3];
			preprocess_mac[i] = new block[total_pre];
			preprocess_key[i] = new block[total_pre];
			sigma_mac[i] = new block[num_ands];
			sigma_key[i] = new block[num_ands];
			eval_labels[i] = new block[cf->num_wire];
		}
		value = new bool[cf->num_wire];
		ANDS_value = new bool[num_ands*3];
		preprocess_value = new bool[total_pre];
		sigma_value = new bool[num_ands];
	}
	~CMPC() {
		if (owns_fpre) delete fpre;
		if(party == 1) {
			delete[] GTM;
			delete[] GTK;
			delete[] GTv;
			delete[] GT;
		}
		delete[] labels;
		for(int i = 1; i <= nP; ++i) {
			delete[] key[i];
			delete[] mac[i];
			delete[] ANDS_key[i];
			delete[] ANDS_mac[i];
			delete[] preprocess_mac[i];
			delete[] preprocess_key[i];
			delete[] sigma_mac[i];
			delete[] sigma_key[i];
			delete[] eval_labels[i];
		}
		delete[] value;
		delete[] ANDS_value;
		delete[] preprocess_value;
		delete[] sigma_value;
	}
	PRG prg;

	void function_independent() {
		if(party != 1)
			prg.random_block(labels, cf->num_wire);

		fpre->compute(ANDS_mac, ANDS_key, ANDS_value, num_ands);
		inst = fpre->inst;

		prg.random_bool(preprocess_value, total_pre);
		fpre->abit->compute(preprocess_mac, preprocess_key, preprocess_value, total_pre);
		auto ret = fpre->abit->check(preprocess_mac, preprocess_key, preprocess_value, total_pre);
		ret.get();

		for(int i = 1; i <= nP; ++i) {
			memcpy(key[i], preprocess_key[i], num_in * sizeof(block));
			memcpy(mac[i], preprocess_mac[i], num_in * sizeof(block));
		}
		memcpy(value, preprocess_value, num_in * sizeof(bool));
#ifdef __debug
		check_MAC<nP>(io, ANDS_mac, ANDS_key, ANDS_value, Delta, num_ands*3, party);
		check_correctness<nP>(io, ANDS_value, num_ands, party);
#endif
//		ret.get();
	}

	void function_dependent() {
		int ands = num_in;
		bool * x[nP+1];
		bool * y[nP+1];
		for(int i = 1; i <= nP; ++i) {
			x[i] = new bool[num_ands];
			y[i] = new bool[num_ands];
		}

		for(int i = 0; i < cf->num_gate; ++i) {
			if (cf->gates[4*i+3] == AND_GATE) {
				for(int j = 1; j <= nP; ++j) {
					key[j][cf->gates[4*i+2]] = preprocess_key[j][ands];
					mac[j][cf->gates[4*i+2]] = preprocess_mac[j][ands];
				}
				value[cf->gates[4*i+2]] = preprocess_value[ands];
				++ands;
			}
		}

		for(int i = 0; i < cf->num_gate; ++i) {
			if (cf->gates[4*i+3] == XOR_GATE) {
				for(int j = 1; j <= nP; ++j) {
					key[j][cf->gates[4*i+2]] = key[j][cf->gates[4*i]] ^ key[j][cf->gates[4*i+1]];
					mac[j][cf->gates[4*i+2]] = mac[j][cf->gates[4*i]] ^ mac[j][cf->gates[4*i+1]];
				}
				value[cf->gates[4*i+2]] = value[cf->gates[4*i]] != value[cf->gates[4*i+1]];
				if(party != 1)
					labels[cf->gates[4*i+2]] = labels[cf->gates[4*i]] ^ labels[cf->gates[4*i+1]];
			} else if (cf->gates[4*i+3] == NOT_GATE) {
				for(int j = 1; j <= nP; ++j) {
					key[j][cf->gates[4*i+2]] = key[j][cf->gates[4*i]];
					mac[j][cf->gates[4*i+2]] = mac[j][cf->gates[4*i]];
				}
				value[cf->gates[4*i+2]] = value[cf->gates[4*i]];
				if(party != 1)
					labels[cf->gates[4*i+2]] = labels[cf->gates[4*i]] ^ Delta;
			}
		}

#ifdef __debug
		check_MAC<nP>(io, mac, key, value, Delta, cf->num_wire, party);
#endif

		ands = 0;
		for(int i = 0; i < cf->num_gate; ++i) {
			if (cf->gates[4*i+3] == AND_GATE) {
				x[party][ands] = value[cf->gates[4*i]] != ANDS_value[3*ands];
				y[party][ands] = value[cf->gates[4*i+1]] != ANDS_value[3*ands+1];	
				ands++;
			}
		}

		// PATCH P4: x and y are opened with their MACs checked (upstream sent bare bits, so a party could flip a
		// gate's output undetected). The MACs travel as one digest per receiver; then the bits are echoed.
		vector<int> ga(num_ands), gb(num_ands);
		ands = 0;
		for(int i = 0; i < cf->num_gate; ++i) if (cf->gates[4*i+3] == AND_GATE) { ga[ands] = cf->gates[4*i]; gb[ands] = cf->gates[4*i+1]; ++ands; }
		vector<future<void>>	 res;
		vector<future<bool>>	 resc;
		for(int i = 1; i <= nP; ++i) for(int j = 1; j <= nP; ++j) if( (i < j) and (i == party or j == party) ) {
			int party2 = i + j - party;
			res.push_back(pool->enqueue([this, x, y, &ga, &gb, party2]() {
				emp::Hash h;
				for(int k = 0; k < num_ands; ++k) { block m = mac[party2][ga[k]] ^ ANDS_mac[party2][3*k]; h.put(&m, sizeof(block)); }
				for(int k = 0; k < num_ands; ++k) { block m = mac[party2][gb[k]] ^ ANDS_mac[party2][3*k+1]; h.put(&m, sizeof(block)); }
				char dg[Hash::DIGEST_SIZE]; h.digest(dg);
				bool * xs = x[party];
				if (num_ands > 0 && g_tamper.hit("xy-share")) { xs = new bool[num_ands]; memcpy(xs, x[party], num_ands); xs[0] = !xs[0]; }
				io->send_data(party2, xs, num_ands);
				io->send_data(party2, y[party], num_ands);
				io->send_data(party2, dg, Hash::DIGEST_SIZE);
				io->flush(party2);
				if (xs != x[party]) delete[] xs;
			}));
			resc.push_back(pool->enqueue([this, x, y, &ga, &gb, party2]() -> bool {
				io->recv_data(party2, x[party2], num_ands);
				io->recv_data(party2, y[party2], num_ands);
				char dg[Hash::DIGEST_SIZE], dg2[Hash::DIGEST_SIZE];
				io->recv_data(party2, dg, Hash::DIGEST_SIZE);
				emp::Hash h;
				for(int k = 0; k < num_ands; ++k) { block m = key[party2][ga[k]] ^ ANDS_key[party2][3*k]; if (x[party2][k]) m = m ^ Delta; h.put(&m, sizeof(block)); }
				for(int k = 0; k < num_ands; ++k) { block m = key[party2][gb[k]] ^ ANDS_key[party2][3*k+1]; if (y[party2][k]) m = m ^ Delta; h.put(&m, sizeof(block)); }
				h.digest(dg2);
				NetIOMP<nP>::note(party2, x[party2], num_ands);
				NetIOMP<nP>::note(party2, y[party2], num_ands);
				return !digest_eq(dg, dg2);
			}));
		}
		joinNclean(res);
		if(joinNcleanCheat(resc)) agmpc_cheat("MAC check of the AND-gate openings x, y failed");
		NetIOMP<nP>::note(party, x[party], num_ands);
		NetIOMP<nP>::note(party, y[party], num_ands);
		io->echo_check("AND-gate openings");
		for(int i = 2; i <= nP; ++i) for(int j = 0; j < num_ands; ++j) {
			x[1][j] = x[1][j] != x[i][j];
			y[1][j] = y[1][j] != y[i][j];
		}

		ands = 0;
		for(int i = 0; i < cf->num_gate; ++i) {
			if (cf->gates[4*i+3] == AND_GATE) {
				for(int j = 1; j <= nP; ++j) {
					sigma_mac[j][ands] = ANDS_mac[j][3*ands+2];
					sigma_key[j][ands] = ANDS_key[j][3*ands+2];
				}
				sigma_value[ands] = ANDS_value[3*ands+2];

				if(x[1][ands]) {
					for(int j = 1; j <= nP; ++j) {
						sigma_mac[j][ands] = sigma_mac[j][ands] ^ ANDS_mac[j][3*ands+1];
						sigma_key[j][ands] = sigma_key[j][ands] ^ ANDS_key[j][3*ands+1];
					}
					sigma_value[ands] = sigma_value[ands] != ANDS_value[3*ands+1];
				}
				if(y[1][ands]) {
					for(int j = 1; j <= nP; ++j) {
						sigma_mac[j][ands] = sigma_mac[j][ands] ^ ANDS_mac[j][3*ands];
						sigma_key[j][ands] = sigma_key[j][ands] ^ ANDS_key[j][3*ands];
					}
					sigma_value[ands] = sigma_value[ands] != ANDS_value[3*ands];
				}
				if(x[1][ands] and y[1][ands]) {
					if(party != 1)
						sigma_key[1][ands] = sigma_key[1][ands] ^ Delta;
					else
						sigma_value[ands] = not sigma_value[ands];
				}
				ands++;
			}
		}//sigma_[] stores the and of input wires to each AND gates
#ifdef __debug_
		check_MAC<nP>(io, sigma_mac, sigma_key, sigma_value, Delta, num_ands, party);
		ands = 0;
		for(int i = 0; i < cf->num_gate; ++i) {
			if (cf->gates[4*i+3] == AND_GATE) {
				bool tmp[] = { value[cf->gates[4*i]], value[cf->gates[4*i+1]], sigma_value[ands]};
				check_correctness(io, tmp, 1, party);
				ands++;
			}
		}
#endif

		ands = 0;
		block H[4][nP+1];
		block K[4][nP+1], M[4][nP+1];
		bool r[4];
		if(party != 1) { 
			for(int i = 0; i < cf->num_gate; ++i) if(cf->gates[4*i+3] == AND_GATE) {
				r[0] = sigma_value[ands] != value[cf->gates[4*i+2]];
				r[1] = r[0] != value[cf->gates[4*i]];
				r[2] = r[0] != value[cf->gates[4*i+1]];
				r[3] = r[1] != value[cf->gates[4*i+1]];

				for(int j = 1; j <= nP; ++j) {
					M[0][j] = sigma_mac[j][ands] ^ mac[j][cf->gates[4*i+2]];
					M[1][j] = M[0][j] ^ mac[j][cf->gates[4*i]];
					M[2][j] = M[0][j] ^ mac[j][cf->gates[4*i+1]];
					M[3][j] = M[1][j] ^ mac[j][cf->gates[4*i+1]];

					K[0][j] = sigma_key[j][ands] ^ key[j][cf->gates[4*i+2]];
					K[1][j] = K[0][j] ^ key[j][cf->gates[4*i]];
					K[2][j] = K[0][j] ^ key[j][cf->gates[4*i+1]];
					K[3][j] = K[1][j] ^ key[j][cf->gates[4*i+1]];
				}
				K[3][1] = K[3][1] ^ Delta;

				Hash(H, labels[cf->gates[4*i]], labels[cf->gates[4*i+1]], ands);
				for(int j = 0; j < 4; ++j) {
					for(int k = 1; k <= nP; ++k) if(k != party) {
						H[j][k] = H[j][k] ^ M[j][k];
						H[j][party] = H[j][party] ^ K[j][k];
					}
					H[j][party] = H[j][party] ^ labels[cf->gates[4*i+2]];
					if(r[j]) 
						H[j][party] = H[j][party] ^ Delta;
				}
				if (ands == 0 && g_tamper.hit("gt-row")) for(int j = 0; j < 4; ++j) H[j][party] = H[j][party] ^ makeBlock(0, 1);
				for(int j = 0; j < 4; ++j)
					io->send_data(1, H[j]+1, sizeof(block)*(nP));
				++ands;
			}
			io->flush(1);
		} else {
			for(int i = 2; i <= nP; ++i) {
				int party2 = i;
				res.push_back(pool->enqueue([this, party2]() {
					for(int i = 0; i < num_ands; ++i)
						for(int j = 0; j < 4; ++j)
							io->recv_data(party2, GT[i][party2][j]+1, sizeof(block)*(nP));
				}));
			}
			for(int i = 0; i < cf->num_gate; ++i) if(cf->gates[4*i+3] == AND_GATE) {
				r[0] = sigma_value[ands] != value[cf->gates[4*i+2]];
				r[1] = r[0] != value[cf->gates[4*i]];
				r[2] = r[0] != value[cf->gates[4*i+1]];
				r[3] = r[1] != value[cf->gates[4*i+1]];
				r[3] = r[3] != true;

				for(int j = 1; j <= nP; ++j) {
					M[0][j] = sigma_mac[j][ands] ^ mac[j][cf->gates[4*i+2]];
					M[1][j] = M[0][j] ^ mac[j][cf->gates[4*i]];
					M[2][j] = M[0][j] ^ mac[j][cf->gates[4*i+1]];
					M[3][j] = M[1][j] ^ mac[j][cf->gates[4*i+1]];

					K[0][j] = sigma_key[j][ands] ^ key[j][cf->gates[4*i+2]];
					K[1][j] = K[0][j] ^ key[j][cf->gates[4*i]];
					K[2][j] = K[0][j] ^ key[j][cf->gates[4*i+1]];
					K[3][j] = K[1][j] ^ key[j][cf->gates[4*i+1]];
				}
				memcpy(GTK[ands], K, sizeof(block)*4*(nP+1));
				memcpy(GTM[ands], M, sizeof(block)*4*(nP+1));
				memcpy(GTv[ands], r, sizeof(bool)*4);
				++ands;
			}
			joinNclean(res);
		}
		for(int i = 1; i <= nP; ++i) {
			delete[] x[i];
			delete[] y[i];
		}
	}

	void online (bool * input, bool * output) {
		bool * mask_input = new bool[cf->num_wire];
		for(int i = 0; i < num_in; ++i)
			mask_input[i] = input[i] != value[i];
		if(party != 1) {
			io->send_data(1, mask_input, num_in);
			io->flush(1);
			io->recv_data(1, mask_input, num_in);
		} else {
			bool * tmp[nP+1];
			for(int i = 2; i <= nP; ++i) tmp[i] = new bool[num_in];
			vector<future<void>> res;
			for(int i = 2; i <= nP; ++i) {
				int party2 = i;
				res.push_back(pool->enqueue([this, tmp, party2]() {
					io->recv_data(party2, tmp[party2], num_in);
				}));
			}
			joinNclean(res);
			for(int i = 0; i < num_in; ++i)
				for(int j = 2; j <= nP; ++j)
					mask_input[i] = tmp[j][i] != mask_input[i];
			for(int i = 2; i <= nP; ++i) {
				int party2 = i;
				res.push_back(pool->enqueue([this, mask_input, party2]() {
					io->send_data(party2, mask_input, num_in);
					io->flush(party2);
				}));
			}
			joinNclean(res);
			for(int i = 2; i <= nP; ++i) delete[] tmp[i];
		}
	
		if(party!= 1) {
			for(int i = 0; i < num_in; ++i) {
				block tmp = labels[i];
				if(mask_input[i]) tmp = tmp ^ Delta;
				io->send_data(1, &tmp, sizeof(block));
			}
			io->flush(1);
		} else {
			vector<future<void>> res;
			for(int i = 2; i <= nP; ++i) {
				int party2 = i;
				res.push_back(pool->enqueue([this, party2]() {
					io->recv_data(party2, eval_labels[party2], num_in*sizeof(block));
				}));
			}
			joinNclean(res);
	
			int ands = 0;	
			for(int i = 0; i < cf->num_gate; ++i) {
				if (cf->gates[4*i+3] == XOR_GATE) {
					for(int j = 2; j<= nP; ++j)
						eval_labels[j][cf->gates[4*i+2]] = eval_labels[j][cf->gates[4*i]] ^ eval_labels[j][cf->gates[4*i+1]];
					mask_input[cf->gates[4*i+2]] = mask_input[cf->gates[4*i]] != mask_input[cf->gates[4*i+1]];
				} else if (cf->gates[4*i+3] == AND_GATE) {
					int index = 2*mask_input[cf->gates[4*i]] + mask_input[cf->gates[4*i+1]];
					block H[nP+1];
					for(int j = 2; j <= nP; ++j)
						eval_labels[j][cf->gates[4*i+2]] = GTM[ands][index][j];
					mask_input[cf->gates[4*i+2]] = GTv[ands][index];
					for(int j = 2; j <= nP; ++j) {
						Hash(H, eval_labels[j][cf->gates[4*i]], eval_labels[j][cf->gates[4*i+1]], ands, index);
						xorBlocks_arr(H, H, GT[ands][j][index], nP+1);
						for(int k = 2; k <= nP; ++k)
							eval_labels[k][cf->gates[4*i+2]] = H[k] ^ eval_labels[k][cf->gates[4*i+2]];
					
						block t0 = GTK[ands][index][j] ^ Delta;

						if(cmpBlock(&H[1], &GTK[ands][index][j], 1))
							mask_input[cf->gates[4*i+2]] = mask_input[cf->gates[4*i+2]] != false;
						else if(cmpBlock(&H[1], &t0, 1))
							mask_input[cf->gates[4*i+2]] = mask_input[cf->gates[4*i+2]] != true;
						else agmpc_cheat("garbled row of AND gate " + std::to_string(ands) + " from " + agmpc_party_name(j) + " matches neither key");
					}
					ands++;
				} else {
					mask_input[cf->gates[4*i+2]] = not mask_input[cf->gates[4*i]];	
					for(int j = 2; j <= nP; ++j)
						eval_labels[j][cf->gates[4*i+2]] = eval_labels[j][cf->gates[4*i]];
				}
			}
		}
		if(party != 1) {
			io->send_data(1, value+cf->num_wire - cf->n3, cf->n3);
			io->flush(1);
		} else {
			vector<future<void>> res;
			bool * tmp[nP+1];
			for(int i = 2; i <= nP; ++i) 
				tmp[i] = new bool[cf->n3];
			for(int i = 2; i <= nP; ++i) {
				int party2 = i;
				res.push_back(pool->enqueue([this, tmp, party2]() {
					io->recv_data(party2, tmp[party2], cf->n3);
				}));
			}
			joinNclean(res);
			for(int i = 0; i < cf->n3; ++i)
				for(int j = 2; j <= nP; ++j)
					mask_input[cf->num_wire - cf->n3 + i] = tmp[j][i] != mask_input[cf->num_wire - cf->n3 + i];
			for(int i = 0; i < cf->n3; ++i)
					mask_input[cf->num_wire - cf->n3 + i] = value[cf->num_wire - cf->n3 + i] != mask_input[cf->num_wire - cf->n3 + i];

			for(int i = 2; i <= nP; ++i) delete[] tmp[i];
			memcpy(output, mask_input + cf->num_wire - cf->n3, cf->n3);
		}
		delete[] mask_input;
	}
	// PATCH P1 (critical): upstream hashed K = sigma(a) ^ sigma^2(b) ^ tweak with fixed-key AES and NO feed-forward,
	// which the evaluator can invert to recover every garbler's Delta (and with it every input). sigma ^ sigma^2 is the
	// identity, so adding feed-forward to that input map is still broken (the opposite row cancels Delta). Here the
	// two labels enter through the GF(2^128) maps 2 and 4 (2, 4, 6 and 3, 5, 7 are all invertible), the tweak is unique
	// per (instance, gate, row, slot), and H(K) = pi(K) ^ K (MMO / random-permutation model; BHKR13's A4 dual-key cipher).
	block hash_tweak(uint64_t idx, uint64_t row, int i) const { return makeBlock((inst << 24) | (4*idx + row), (uint64_t)i); }
	void Hash(block H[4][nP+1], const block & a, const block & b, uint64_t idx) {
		const block A = gf128_double(a), B = gf128_double(gf128_double(b));
		block K0[4];
		K0[0] = A ^ B;
		K0[1] = K0[0] ^ Delta4;           // row 1: b ^ Delta
		K0[2] = K0[0] ^ Delta2;           // row 2: a ^ Delta
		K0[3] = K0[0] ^ Delta2 ^ Delta4;  // row 3: both
		for(int j = 0; j < 4; ++j) {
			for(int i = 1; i <= nP; ++i) H[j][i] = K0[j] ^ hash_tweak(idx, j, i);
			block P[nP+1];
			memcpy(P+1, H[j]+1, sizeof(block)*nP);
			prp.permute_block(P+1, nP);
			for(int i = 1; i <= nP; ++i) H[j][i] = H[j][i] ^ P[i];
		}
	}

	void Hash(block H[nP+1], const block &a, const block& b, uint64_t idx, uint64_t row) {
		const block K0 = gf128_double(a) ^ gf128_double(gf128_double(b));
		for(int i = 1; i <= nP; ++i) H[i] = K0 ^ hash_tweak(idx, row, i);
		block P[nP+1];
		memcpy(P+1, H+1, sizeof(block)*nP);
		prp.permute_block(P+1, nP);
		for(int i = 1; i <= nP; ++i) H[i] = H[i] ^ P[i];
	}

	string tostring(bool a) {
		if(a) return "T";
		else return "F";
	}

	void online (bool * input, bool * output, int* start, int* end) {
		bool * mask_input = new bool[cf->num_wire];
		bool * input_mask[nP+1];
		for(int i = 0; i <= nP; ++i) input_mask[i] = new bool[end[party] - start[party]];
		memcpy(input_mask[party], value+start[party], end[party] - start[party]);
		memcpy(input_mask[0], input+start[party], end[party] - start[party]);

		vector<future<bool>> res;
		for(int i = 1; i <= nP; ++i) for(int j = 1; j<= nP; ++j) if( (i < j) and (i == party or j == party) ) {
			int party2 = i + j - party;
			res.push_back(pool->enqueue([this, start, end, party2]() {
				char dig[Hash::DIGEST_SIZE];
				io->send_data(party2, value+start[party2], end[party2]-start[party2]);
				emp::Hash::hash_once(dig, mac[party2]+start[party2], (end[party2]-start[party2])*sizeof(block));
				io->send_data(party2, dig, Hash::DIGEST_SIZE);
				io->flush(party2);
				return false;
			}));
			res.push_back(pool->enqueue([this, start, end, input_mask, party2]() {
				char dig[Hash::DIGEST_SIZE];
				char dig2[Hash::DIGEST_SIZE];
				io->recv_data(party2, input_mask[party2], end[party]-start[party]);
				block * tmp = new block[end[party]-start[party]];
				for(int i =  0; i < end[party] - start[party]; ++i) {
					tmp[i] = key[party2][i+start[party]];
					if(input_mask[party2][i])tmp[i] = tmp[i] ^ Delta;
				}
				emp::Hash::hash_once(dig2, tmp, (end[party]-start[party])*sizeof(block));
				io->recv_data(party2, dig, Hash::DIGEST_SIZE);
				delete[] tmp;
				return !digest_eq(dig, dig2);	
			}));
		}
		if(joinNcleanCheat(res)) agmpc_cheat("cheat!");
		for(int i = 1; i <= nP; ++i)
			for(int j = 0; j < end[party] - start[party]; ++j)
				input_mask[0][j] = input_mask[0][j] != input_mask[i][j];


		if(party != 1) {
			io->send_data(1, input_mask[0], end[party] - start[party]);
			io->flush(1);
			io->recv_data(1, mask_input, num_in);
		} else {
			vector<future<void>> res;
			for(int i = 2; i <= nP; ++i) {
				int party2 = i;
				res.push_back(pool->enqueue([this, mask_input, start, end , party2]() {
					io->recv_data(party2, mask_input+start[party2], end[party2] - start[party2]);
				}));
			}
			joinNclean(res);
			memcpy(mask_input, input_mask[0], end[1]-start[1]);
			for(int i = 2; i <= nP; ++i) {
				int party2 = i;
				res.push_back(pool->enqueue([this, mask_input, party2]() {
					io->send_data(party2, mask_input, num_in);
					io->flush(party2);
				}));
			}
			joinNclean(res);
		}
	
		if(party!= 1) {
			for(int i = 0; i < num_in; ++i) {
				block tmp = labels[i];
				if(mask_input[i]) tmp = tmp ^ Delta;
				io->send_data(1, &tmp, sizeof(block));
			}
			io->flush(1);
		} else {
			vector<future<void>> res;
			for(int i = 2; i <= nP; ++i) {
				int party2 = i;
				res.push_back(pool->enqueue([this, party2]() {
					io->recv_data(party2, eval_labels[party2], num_in*sizeof(block));
				}));
			}
			joinNclean(res);
	
			int ands = 0;	
			for(int i = 0; i < cf->num_gate; ++i) {
				if (cf->gates[4*i+3] == XOR_GATE) {
					for(int j = 2; j<= nP; ++j)
						eval_labels[j][cf->gates[4*i+2]] = eval_labels[j][cf->gates[4*i]] ^ eval_labels[j][cf->gates[4*i+1]];
					mask_input[cf->gates[4*i+2]] = mask_input[cf->gates[4*i]] != mask_input[cf->gates[4*i+1]];
				} else if (cf->gates[4*i+3] == AND_GATE) {
					int index = 2*mask_input[cf->gates[4*i]] + mask_input[cf->gates[4*i+1]];
					block H[nP+1];
					for(int j = 2; j <= nP; ++j)
						eval_labels[j][cf->gates[4*i+2]] = GTM[ands][index][j];
					mask_input[cf->gates[4*i+2]] = GTv[ands][index];
					for(int j = 2; j <= nP; ++j) {
						Hash(H, eval_labels[j][cf->gates[4*i]], eval_labels[j][cf->gates[4*i+1]], ands, index);
						xorBlocks_arr(H, H, GT[ands][j][index], nP+1);
						for(int k = 2; k <= nP; ++k)
							eval_labels[k][cf->gates[4*i+2]] = H[k] ^ eval_labels[k][cf->gates[4*i+2]];
					
						block t0 = GTK[ands][index][j] ^ Delta;

						if(cmpBlock(&H[1], &GTK[ands][index][j], 1))
							mask_input[cf->gates[4*i+2]] = mask_input[cf->gates[4*i+2]] != false;
						else if(cmpBlock(&H[1], &t0, 1))
							mask_input[cf->gates[4*i+2]] = mask_input[cf->gates[4*i+2]] != true;
						else agmpc_cheat("garbled row of AND gate " + std::to_string(ands) + " from " + agmpc_party_name(j) + " matches neither key");
					}
					ands++;
				} else {
					mask_input[cf->gates[4*i+2]] = not mask_input[cf->gates[4*i]];	
					for(int j = 2; j <= nP; ++j)
						eval_labels[j][cf->gates[4*i+2]] = eval_labels[j][cf->gates[4*i]];
				}
			}
		}
		if(party != 1) {
			io->send_data(1, value+cf->num_wire - cf->n3, cf->n3);
			io->flush(1);
		} else {
			vector<future<void>> res;
			bool * tmp[nP+1];
			for(int i = 2; i <= nP; ++i) 
				tmp[i] = new bool[cf->n3];
			for(int i = 2; i <= nP; ++i) {
				int party2 = i;
				res.push_back(pool->enqueue([this, tmp, party2]() {
					io->recv_data(party2, tmp[party2], cf->n3);
				}));
			}
			joinNclean(res);
			for(int i = 0; i < cf->n3; ++i)
				for(int j = 2; j <= nP; ++j)
					mask_input[cf->num_wire - cf->n3 + i] = tmp[j][i] != mask_input[cf->num_wire - cf->n3 + i];
			for(int i = 0; i < cf->n3; ++i)
					mask_input[cf->num_wire - cf->n3 + i] = value[cf->num_wire - cf->n3 + i] != mask_input[cf->num_wire - cf->n3 + i];

			for(int i = 2; i <= nP; ++i) delete[] tmp[i];
			memcpy(output, mask_input + cf->num_wire - cf->n3, cf->n3);
		}
		delete[] mask_input;
	}

	void online (FlexIn<nP> * input, FlexOut<nP> *output) {
		bool * mask_input = new bool[cf->num_wire];
		input->associate_cmpc(pool, value, mac, key, io, Delta);
		input->input(mask_input);

		if(party!= 1) {
			const bool bad_label = num_in > 0 && g_tamper.hit("label");
			for(int i = 0; i < num_in; ++i) {
				block tmp = labels[i];
				if(mask_input[i]) tmp = tmp ^ Delta;
				if(i == 0 && bad_label) tmp = tmp ^ makeBlock(0, 1);
				io->send_data(1, &tmp, sizeof(block));
			}
			io->flush(1);
			agmpc_phase("online/evaluate");
		} else {
			vector<future<void>> res;
			for(int i = 2; i <= nP; ++i) {
				int party2 = i;
				res.push_back(pool->enqueue([this, party2]() {
					io->recv_data(party2, eval_labels[party2], num_in*sizeof(block));
				}));
			}
			joinNclean(res);
			agmpc_phase("online/evaluate");

			int ands = 0;
			for(int i = 0; i < cf->num_gate; ++i) {
				if (cf->gates[4*i+3] == XOR_GATE) {
					for(int j = 2; j<= nP; ++j)
						eval_labels[j][cf->gates[4*i+2]] = eval_labels[j][cf->gates[4*i]] ^ eval_labels[j][cf->gates[4*i+1]];
					mask_input[cf->gates[4*i+2]] = mask_input[cf->gates[4*i]] != mask_input[cf->gates[4*i+1]];
				} else if (cf->gates[4*i+3] == AND_GATE) {
					int index = 2*mask_input[cf->gates[4*i]] + mask_input[cf->gates[4*i+1]];
					block H[nP+1];
					for(int j = 2; j <= nP; ++j)
						eval_labels[j][cf->gates[4*i+2]] = GTM[ands][index][j];
					mask_input[cf->gates[4*i+2]] = GTv[ands][index];
					for(int j = 2; j <= nP; ++j) {
						Hash(H, eval_labels[j][cf->gates[4*i]], eval_labels[j][cf->gates[4*i+1]], ands, index);
						xorBlocks_arr(H, H, GT[ands][j][index], nP+1);
						for(int k = 2; k <= nP; ++k)
							eval_labels[k][cf->gates[4*i+2]] = H[k] ^ eval_labels[k][cf->gates[4*i+2]];

						block t0 = GTK[ands][index][j] ^ Delta;

						if(cmpBlock(&H[1], &GTK[ands][index][j], 1))
							mask_input[cf->gates[4*i+2]] = mask_input[cf->gates[4*i+2]] != false;
						else if(cmpBlock(&H[1], &t0, 1))
							mask_input[cf->gates[4*i+2]] = mask_input[cf->gates[4*i+2]] != true;
						else agmpc_cheat("garbled row of AND gate " + std::to_string(ands) + " from " + agmpc_party_name(j) + " matches neither key");
					}
					ands++;
				} else {
					mask_input[cf->gates[4*i+2]] = not mask_input[cf->gates[4*i]];
					for(int j = 2; j <= nP; ++j)
						eval_labels[j][cf->gates[4*i+2]] = eval_labels[j][cf->gates[4*i]];
				}
			}
		}

		agmpc_phase("online/output");
		output->associate_cmpc(pool, value, mac, key, eval_labels, labels, io, Delta);
		output->output(mask_input, cf->num_wire - cf->n3);

		delete[] mask_input;
	}
};
#endif// CMPC_H__
