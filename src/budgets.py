from amplification.counter import Counter1, Counter2

eps_ld = 100
dim = 50890 # 197258
m = 1000
c1 = Counter1(dim=dim, m=m, e_l=eps_ld)
c2 = Counter2(rate=50, m_p=int(m/3), dim=dim, m=m, e_l=eps_ld)
print("-------------Counter1-------------")
c1.print_details()
print("-------------Counter2-------------")
c2.print_details()
c2.no_sub_amplification()
