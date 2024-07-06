# --- EDI-3 ---

# # PALE
# python -m test.homology_degree -s data/edi3/age1.txt -t data/edi3/age2.txt -c cfgs/pale_edi3.yaml --num_exps 10
# python -m test.homology_degree -s data/edi3/age1.txt -t data/edi3/age4.txt -c cfgs/pale_edi3.yaml --num_exps 10
# python -m test.homology_degree -s data/edi3/an.txt -t data/edi3/bn.txt -c cfgs/pale_edi3.yaml --num_exps 10

# SHELLEY
python -m test.homology_degree -s data/edi3/age1.txt -t data/edi3/age2.txt -c cfgs/shelley_share-gine2-sgm_edi3.yaml --num_exps 10
python -m test.homology_degree -s data/edi3/age1.txt -t data/edi3/age4.txt -c cfgs/shelley_share-gine2-sgm_edi3.yaml --num_exps 10
python -m test.homology_degree -s data/edi3/an.txt -t data/edi3/bn.txt -c cfgs/shelley_share-gine2-sgm_edi3.yaml --num_exps 10

# # # FINAL
# python -m test.homology_degree -s data/edi3/age1.txt -t data/edi3/age2.txt -c cfgs/final.yaml --num_exps 10
# python -m test.homology_degree -s data/edi3/age1.txt -t data/edi3/age4.txt -c cfgs/final.yaml --num_exps 10
# python -m test.homology_degree -s data/edi3/an.txt -t data/edi3/bn.txt -c cfgs/final.yaml --num_exps 10

# # MAGNA
# python -m test.homology_degree -s data/edi3/age1.txt -t data/edi3/age2.txt -c cfgs/magna_edi3.yaml --num_exps 3
# python -m test.homology_degree -s data/edi3/age1.txt -t data/edi3/age4.txt -c cfgs/magna_edi3.yaml --num_exps 3
# python -m test.homology_degree -s data/edi3/an.txt -t data/edi3/bn.txt -c cfgs/magna_edi3.yaml --num_exps 3

# --- BN1000 ---

# # PALE
# python -m test.homology_degree -s data/bn1000/ad1.txt -t data/bn1000/ad2.txt -c cfgs/pale_bn.yaml --num_exps 10
# python -m test.homology_degree -s data/bn1000/ad1.txt -t data/bn1000/ne1.txt -c cfgs/pale_bn.yaml --num_exps 10
# python -m test.homology_degree -s data/bn1000/ne1.txt -t data/bn1000/6m1.txt -c cfgs/pale_bn.yaml --num_exps 10

# SHELLEY
python -m test.homology_degree -s data/bn1000/ad1.txt -t data/bn1000/ad2.txt -c cfgs/shelley_share-gine2-sgm_bn.yaml  --num_exps 10
python -m test.homology_degree -s data/bn1000/ad1.txt -t data/bn1000/ne1.txt -c cfgs/shelley_share-gine2-sgm_bn.yaml  --num_exps 10
python -m test.homology_degree -s data/bn1000/ne1.txt -t data/bn1000/6m1.txt -c cfgs/shelley_share-gine2-sgm_bn.yaml  --num_exps 10

# # FINAL
# python -m test.homology_degree -s data/bn1000/ad1.txt -t data/bn1000/ad2.txt -c cfgs/final.yaml --num_exps 10
# python -m test.homology_degree -s data/bn1000/ad1.txt -t data/bn1000/ne1.txt -c cfgs/final.yaml --num_exps 10
# python -m test.homology_degree -s data/bn1000/ne1.txt -t data/bn1000/6m1.txt -c cfgs/final.yaml --num_exps 10

# MAGNA
# python -m test.homology_degree -s data/bn1000/ad1.txt -t data/bn1000/ad2.txt -c cfgs/final.yaml -c cfgs/magna_bn.yaml --num_exps 3
# python -m test.homology_degree -s data/bn1000/ad1.txt -t data/bn1000/ne1.txt -c cfgs/final.yaml -c cfgs/magna_bn.yaml --num_exps 3
# python -m test.homology_degree -s data/bn1000/ne1.txt -t data/bn1000/6m1.txt -c cfgs/final.yaml -c cfgs/magna_bn.yaml --num_exps 3