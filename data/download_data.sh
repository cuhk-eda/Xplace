rm -rf raw/ cad/
mkdir raw

echo "=== Downloading ispd2005 ==="
wget --no-check-certificate "https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136644_link_cuhk_edu_hk/EQOdYYk6xgxLq4MRyST7keIBPbmphF1_0yUDDzv1g3JBRA?e=c6xjxV&download=1" -O ispd2005.tar.gz
tar xvzf ispd2005.tar.gz
rm -rf ispd2005.tar.gz
mv ispd2005/ raw/

echo "=== Downloading ispd2015 ==="
wget --no-check-certificate "https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136644_link_cuhk_edu_hk/Ea4YjKNvi-9CnekS41Pw-GgBEhIRNnp6AhMDU9_xElLjNA?e=YSUMhQ&download=1" -O ispd2015.tar.gz
tar xvzf ispd2015.tar.gz
rm -rf ispd2015.tar.gz
mv ispd2015/ raw/
echo "=== Preprocessing ispd2015 to generate ispd2015_fix ==="
python fix_ispd2015_route.py

echo "=== Downloading iccad2019 ==="
wget --no-check-certificate "https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136644_link_cuhk_edu_hk/EYdqZNU7EGtEos77fGUSUkkBMdCOETzGm-Dws1XNnr_9BQ?e=3HYQLO&download=1" -O iccad2019.tar.gz
tar xvzf iccad2019.tar.gz
rm -rf iccad2019.tar.gz
mv iccad2019/ raw/

# echo "=== (Optional) Converting raw design to torch data ==="
# python convert_design_to_torch_data.py --dataset ispd2005
# python convert_design_to_torch_data.py --dataset ispd2015_fix
# python convert_design_to_torch_data.py --dataset iccad2019