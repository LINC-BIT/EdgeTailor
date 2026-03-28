import itertools

def heuristic_search(SL, SH, eval_func, T, M, T_avail, M_avail, K=50, delta_q_list=[1, 2, 5]):

    L, H, q = None, None, 1
    best_acc, TL, ML = eval_func(L, H, q)

    for round_idx in range(K):
        candidates = []


        for Lp in SL:
            param_dict = SH.get(Lp, {})
            if not param_dict:
                continue


            keys, values = zip(*param_dict.items())
            for combo in itertools.product(*values):
                Hp = dict(zip(keys, combo))
                acc_new, TLp, MLp = eval_func(Lp, Hp, q)
                delta_A = acc_new - best_acc
                delta_T = q * (TLp - (TL if L else 0))
                delta_M = MLp - (ML if L else 0)

                if delta_T <= T_avail and delta_M <= M_avail:
                    reward = delta_A / (delta_T + 1e-8) if delta_T > 0 else 0
                    candidates.append((Lp, Hp, q, acc_new, delta_A, delta_T, delta_M, reward))


        for dq in delta_q_list:
            acc_new, TL_cur, ML_cur = eval_func(L, H, q + dq)
            delta_A = acc_new - best_acc
            delta_T = dq * (T + (TL if L else 0))
            delta_M = 0

            if delta_T <= T_avail:
                reward = delta_A / (delta_T + 1e-8)
                candidates.append((L, H, q + dq, acc_new, delta_A, delta_T, delta_M, reward))

        if not candidates:
            break


        Lp, Hp, qp, acc_new, delta_A, delta_T, delta_M, reward = max(candidates, key=lambda x: x[-1])


        L, H, q = Lp, Hp, qp
        best_acc = acc_new
        T_avail -= delta_T
        M_avail -= delta_M
        TL = eval_func(L, H, q)[1] if L else 0
        ML = eval_func(L, H, q)[2] if L else 0

        if T_avail <= 0:
            break

    return L, H, q, best_acc