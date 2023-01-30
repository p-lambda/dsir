set -x

mkdir -p logs/qualitystats

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "${parent_path}"
# load global parameters: CACHE and ROOT_DIR
source ../config.sh

awk 1 ${PILE_PATH}/chunked/01_128/01_128.json ${PILE_PATH}/chunked/02_128/02_128.json ${PILE_PATH}/chunked/03_128/03_128.json ${PILE_PATH}/chunked/04_128/04_128.json ${PILE_PATH}/chunked/05_128/05_128.json ${PILE_PATH}/chunked/06_128/06_128.json ${PILE_PATH}/chunked/07_128/07_128.json ${PILE_PATH}/chunked/08_128/08_128.json ${PILE_PATH}/chunked/09_128/09_128.json ${PILE_PATH}/chunked/10_128/10_128.json ${PILE_PATH}/chunked/11_128/11_128.json
${PILE_PATH}/chunked/12_128/12_128.json ${PILE_PATH}/chunked/13_128/13_128.json ${PILE_PATH}/chunked/14_128/14_128.json ${PILE_PATH}/chunked/15_128/15_128.json ${PILE_PATH}/chunked/16_128/16_128.json ${PILE_PATH}/chunked/17_128/17_128.json ${PILE_PATH}/chunked/18_128/18_128.json ${PILE_PATH}/chunked/19_128/19_128.json ${PILE_PATH}/chunked/20_128/20_128.json ${PILE_PATH}/chunked/21_128/21_128.json ${PILE_PATH}/chunked/22_128/22_128.json
${PILE_PATH}/chunked/23_128/23_128.json ${PILE_PATH}/chunked/24_128/24_128.json ${PILE_PATH}/chunked/25_128/25_128.json ${PILE_PATH}/chunked/26_128/26_128.json ${PILE_PATH}/chunked/27_128/27_128.json ${PILE_PATH}/chunked/28_128/28_128.json ${PILE_PATH}/chunked/29_128/29_128.json > ${CACHE}/combined_all.json

mv ${CACHE}/combined_all.json ${PILE_PATH}/chunked/combined_all.json
