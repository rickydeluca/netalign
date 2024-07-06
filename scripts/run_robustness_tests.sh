# python -m test.robustness -s data/edi3/edi3.txt -c cfgs/isorank.yaml
# python -m test.robustness -s data/edi3/edi3.txt -c cfgs/bigalign.yaml
# python -m test.robustness -s data/edi3/edi3.txt -c cfgs/final.yaml
# python -m test.robustness -s data/edi3/edi3.txt -c cfgs/ione.yaml
# python -m test.robustness -s data/edi3/edi3.txt -c cfgs/magna_edi3.yaml
# python -m test.robustness -s data/edi3/edi3.txt -c cfgs/pale_edi3.yaml
# python -m test.robustness -s data/edi3/edi3.txt -c cfgs/shelley_share-gine2-sgm_edi3.yaml
# python -m test.robustness -s data/edi3/edi3.txt -c cfgs/shelley_degree-gine2-sigma_edi3.yaml

# python -m test.robustness -s data/bn1000/ad1.txt -c cfgs/isorank.yaml
# python -m test.robustness -s data/bn1000/ad1.txt -c cfgs/bigalign.yaml
# python -m test.robustness -s data/bn1000/ad1.txt -c cfgs/final.yaml
# python -m test.robustness -s data/bn1000/ad1.txt -c cfgs/ione.yaml
python -m test.robustness -s data/bn1000/ad1.txt -c cfgs/magna_bn.yaml --size 3   # TODO
# python -m test.robustness -s data/bn1000/ad1.txt -c cfgs/pale_bn.yaml
# python -m test.robustness -s data/bn1000/ad1.txt -c cfgs/shelley_share-gine2-sgm_bn.yaml --size 3
# python -m test.robustness -s data/bn1000/ad1.txt -c cfgs/shelley_degree-gine2-sigma_bn.yaml --size 3

# python -m test.robustness -s data/ppi/ppi.txt -c cfgs/isorank.yaml
# python -m test.robustness -s data/ppi/ppi.txt -c cfgs/bigalign.yaml
# python -m test.robustness -s data/ppi/ppi.txt -c cfgs/final.yaml
# python -m test.robustness -s data/ppi/ppi.txt -c cfgs/ione.yaml
python -m test.robustness -s data/ppi/ppi.txt -c cfgs/magna_ppi.yaml --size 3 # TODO
# python -m test.robustness -s data/ppi/ppi.txt -c cfgs/pale_ppi.yaml
# python -m test.robustness -s data/ppi/ppi.txt -c cfgs/shelley_share-gine2-sgm_ppi.yaml --size 3
# python -m test.robustness -s data/ppi/ppi.txt -c cfgs/shelley_degree-gine2-sigma_ppi.yaml --size 3

# python -m test.robustness -s data/edi3/edi3.txt -c cfgs/deeplink.yaml --size 3
# python -m test.robustness -s data/bn1000/ad1.txt -c cfgs/deeplink.yaml --size 3
# python -m test.robustness -s data/ppi/ppi.txt -c cfgs/deeplink.yaml --size 3
