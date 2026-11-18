"""
Compare the backache dataset from PMLB against the OCR'd Table D.3 from
Chatfield's Problem Solving (2nd ed, 1995).

Parses the OCR text, loads the PMLB CSV, and reports all discrepancies.
"""

import csv
import re
import sys

# ============================================================================
# Parse OCR text from Chatfield's Table D.3
# ============================================================================

# The raw OCR text pasted from the book. Each data line has the format:
#   ID severity month age height preWeight pregWeight babyWeight PACKED_STRING
# where PACKED_STRING = parity(1) + prevBackache(1) + relief1-8(8) + aggrav1-15(15) = 25 chars

RAW_OCR = r"""
001 1 0 26 1.52 54.5 75.0 3.35 0100000000000000000000000
002 3 0 23 1.60 59.1 68.6 2.22 1210010000001001000000000
003 2 6 24 1.57 73.2 82.7 4.15 0100011010100301000000000
004 1 8 22 1,52 41.4 47.3 2.81 0100000000000001000000000
005 1 0 27 1.60 55.5 60.0 3.75 L200000000000000000000000
006 1 7 32 1,75 70.5 85.5 4,01 2400000000000001000000010
007 1 0 24 1.73 76.4 89.1 3.41 0100000000000000000000000
008 1 8 25 1.63 70.0 85.0 4.01 1100010000100000000000000
009 2 6 20 1,55 52.3 59.5 3.69 1400000010000000000001000
010 2 8 18 1.63 83.2 90.9 3.30 0110000100000001000000010
011 1 0 21 1.65 64.5 75.5 2,95 0100000000000000000000000
012 1 0 26 1.55 49.5 53.6 2.64 0110000000100000000000000
013 2 6 35 1.65 70.0 82.7 3.64 7200100000010000000000100
014 1 8 26 1,60 52.3 64,5 4.49 1300110001OOOOOOQOOOOO L 00
015 1 6 34 1.68 68.2 77.3 3.75 3300010010300001000001000
016 0 0 25 1.50 47.3 55.0 2.73 1000000000000000000000000
017 1 7 42 1.52 66,8 73.2 2.44 6400010000010000000000000
018 2 6 26 1.65 70.0 81.4 3.01 1300010000000101000000010
019 2 6 18 1.60 56.4 70.0 3.89 1300010100000001000001001
020 0 1 42 1,65 53,6 63.6 2.73 2000000000000000000000000
021 2 0 28 1,63 59.1 72.3 3.75 1300000000000011000001001
022 1 0 26 1.52 44.5 56.4 3.49 0101000000000100000000000
023 1 0 23 1.57 55.9 60.9 3.07 01OOOOOOOOOOOOOOOOOOOOOOQ
024 2 6 21 1.55 57.3 77.3 3.35 0100010000000000000001001
025 2 7 32 1.52 69.5 75.5 3.64 5300100010010101000000000
026 1 8 18 1.60 73.2 81.4 2.05 0100001000010001000010000
027 1 0 25 1.70 52.3 59.5 2.44 1310000000000101010010000
028 1 0 30 1.63 62.7 72.3 3.07 400QOOOOOOO(X)OOOOOOOOOOOQ
029 1 0 19 1.65 73.6 92.7 3.35 0000000000000000000000000
030 2 7 26 1.65 70.0 89.1 3.21 1200000000111111000000000
031 1 8 28 1.68 56.8 70.9 3.41 0100000000000000000001000
032 1 0 21 1.60 58.2 69.5 3.30 OOOOOOOOOOO11111000001101
033 0 0 29 1.57 68.2 75.0 3.35 CRKXMOOOOOOOOOOOOOOOOOOOO
034 1 8 27 1.65 50.9 66.4 3.10 1200010000000000000000100
035 2 2 30 1.60 50.9 62.7 3.75 1210000000000000000001000
036 2 5 26 1.75 69.5 84.1 4.20 01OOO10000000000000001000
037 1 8 21 1.60 62.7 73.2 2.95 0100010000000000000000001
038 2 2 24 1.73 63.6 69.5 3.18 1200000100111001000001000
039 2 8 28 1.50 55.5 66.4 4.35 2400000000000000000000000
040 1 8 27 1.57 55.9 62.3 3.69 0100110000010100000000000
041 3 6 32 1.60 47.7 66.8 3.41 5310100000100100001000001
042 3 5 26 1.63 55.5 63.2 3.30 0100011001100101000000110
043 2 7 37 1.63 60,9 69.5 3.41 2200001010000000000000100
044 3 7 31 1.60 60.5 77.3 3.27 1000100100000000000000000
045 1 0 24 1.57 62.3 77.3 3.38 0000000000000000000000000
046 3 5 24 1.65 57.3 73.6 3.69 0000010100001101000001001
047 1 0 23 1.52 55.5 69.5 3.35 0100000000000000000000000
048 2 3 31 1.60 47.7 55.9 1.62 0110000010100000000000000
049 1 8 37 1.60 76,4 891 3,58 1300010000010100000000000
050 2 6 23 1.65 95.5 95.5 2.73 1200010010000000000001011
051 1 4 39 1.70 72.3 89.1 3.44 0100010000100000000000100
052 2 6 32 1.70 65.9 75,0 3,66 1400000000000000000001110
053 2 4 30 1.52 50.9 57.3 2.56 1301101000101111000001000
054 2 3 24 3.57 57,3 58.6 1,08 0100010100000000000000000
055 1 0 24 1.75 69.1 77.3 3.38 0100000000000000000000000
056 2 6 29 1.55 47.7 54.5 3.81 2300000010000000000001000
057 1 0 24 1.75 59.1 68.6 2.98 3200000000000000000000000
058 2 7 27 1.70 60.0 71.4 2.95 1210010010010100001100100
059 2 4 30 1.57 60.0 75.9 2.84 1201000010000000000001000
060 2 3 32 1.63 66.4 82.7 2.64 2310011110000100000001110
061 1 0 23 1.55 44.5 48.2 2.27 0100000000000000000000000
062 1 0 18 1.60 63.6 75.0 3.47 0100000000000000000000000
063 2 9 28 1.55 47.7 58.2 2.56 1200010000100001000001000
064 1 0 24 1.50 45.9 55.0 3.10 1400000001000100000000000
065 1 5 22 1.68 53.2 64.5 3.07 0100000010001100000000001
066 2 5 29 1.57 60.5 74.1 3.27 1300010100000100000000000
067 3 2 30 1.65 89.5 97.7 3.81 1301000000100010000000000
068 2 4 22 1.70 61.8 75.5 3.21 0111000010111100000001000
069 2 5 31 1.63 66.4 82.3 2.27 0100010000000001000000000
070 1 0 29 1.57 60.5 64.1 3.30 3200000000000000000000000
071 1 0 28 1.65 60.5 73.2 3.75 01ooooooooooooooooooooooo
072 3 6 32 1.60 47.7 50.5 1.90 3300110000010000000001100
073 1 7 21 1.55 67.3 80.0 3.27 0100000000100000000000000
074 2 5 24 1.50 50.0 63.6 2.76 0100010000100000000000000
075 2 5 26 1.65 55.5 67.3 4,15 2400010100101011000001000
076 1 0 26 1.73 57.3 65.5 3.07 1200000000010000000000000
077 2 7 26 1.60 65.0 80.5 3.95 0100000000000100000000000
078 3 0 28 1.65 57.3 82.7 3.89 0100010010110011000001100
079 2 0 36 1.73 75.9 87.3 3.64 1300010010111111000000001
080 2 7 29 1.55 75.5 78.6 2.61 0100000010101000000010000
081 1 7 23 1.65 47.7 51.8 2.84 1300010000000101000001001
082 3 8 19 1.70 57.3 74.5 3.69 2201110100010010000000001
083 3 7 30 1.73 63.6 80.5 3.61 0410010010001000000001000
084 2 6 24 1.60 50.0 59.1 3,52 1200110100111100000000000
085 1 0 24 1.70 60.0 65.0 3.18 0100000000100000000000000
086 1 0 30 1.55 47.7 59.1 3.15 1200000000000000000000000
087 2 5 25 1.65 50.5 60.9 3.35 1200010000001000000000000
088 3 7 23 1.55 62,7 85.5 4.18 0100010000011111000001001
089 3 3 33 1.63 63.6 70.9 2.93 1310010010100001000001000
090 1 0 31 1.52 52.3 53.2 2.39 2200000000000000000000000
091 2 7 21 1.50 63.6 63.6 2.98 2200000000000000000000000
092 1 0 27 1.75 68.2 82.7 5.97 00000000001oooooooooooooo
093 1 5 25 1.60 68.2 76.4 3.18 0100010000000000000000100
094 3 2 34 1.63 66.8 80.5 4.15 3210000000000000000000101
095 2 6 26 1.70 65.9 68.2 3.35 1400000000100000000000000
096 0 0 20 1.57 60.5 69.5 3.24 0000000000000000000000000
097 2 0 28 1.68 89.3 94.5 2.59 0100010000000101000000000
098 2 0 26 1.70 80.5 89.1 1.42 0010010010111111000001001
099 2 6 22 1.60 63.6 82.7 3.89 2300100000000000000001000
100 2 4 26 1.50 43.6 53.6 3.81 0100000010000000000000101
101 1 8 21 1.55 54.5 67.3 3.72 0100010100000000000000100
102 1 7 18 1.60 62.7 71.4 2.44 1oooooooooooooooooooooooo
103 2 6 22 1.52 54.1 65.0 2.78 0101000010000000000001010
104 0 0 20 1.63 53.6 73.2 2.78 0000000000000000000000000
105 1 0 25 1.65 61.8 66.8 3.18 01ooooooooooooooooooooooo
106 2 4 23 1.57 76,4 87.3 2.78 1100010000001000000000100
107 1 5 22 1.60 74.5 82.7 2.90 0100000010000000000010000
108 2 6 22 1.68 47.7 53.6 3.01 3300010000010000000001000
109 1 8 32 1.63 52.3 56.8 2.50 01000000101oooooooooooooo
110 2 6 26 1.57 53.6 65.0 2.87 I200010000000000100000000
111 1 0 25 1.63 58.6 62.3 2.44 0100000000000000000000000
112 0 1 19 1.57 51.8 71.8 3.35 0000000000000000000000000
113 1 0 21 1.60 76.4 78.2 3.07 01ooooooooooooooooooooooo
114 1 0 23 1.60 56.4 71.4 2.50 0Iooooooooooooooooooooooo
115 2 7 20 1.60 54.1 68.2 3.58 0100010001000101000000101
116 1 4 19 1.65 57.3 70.0 2.61 1100100000000000000000000
117 2 8 30 1.50 53.2 66.4 3.98 1300100100010100000001000
118 1 0 26 1.60 57.3 65.9 3.89 OOOOOOOOOOOOOOOOOOOOOOO10
119 1 0 24 1.57 60.5 71.5 3.24 01ooooooooooooooooooooooo
120 1 6 28 1.60 46.8 60.5 2.98 1300000010100000000000001
121 0 0 16 1.63 60.5 65.9 3.18 0000O0(X)000O00CK)000O00000
122 1 0 29 1.63 54.5 67.3 3.58 0100000000000000000000000
123 3 6 25 1.57 59.1 69.5 3.89 1200000010101000000011001
124 2 3 17 1.52 48.2 61.8 2.78 0000100010001000011100000
125 1 7 23 1.65 77.3 83.2 3.58 0100000010000000000000100
126 1 0 26 1.63 72.7 80.0 2.95 01ooooooooooooooooooooooo
127 1 0 20 1.63 58.6 70.0 2.84 1100000100000000000000000
128 2 2 25 1.68 54.5 71.8 3.72 0100010010001101000000000
129 1 3 24 1.63 61.8 73.6 2.84 0100000010000000000001110
130 3 3 35 1.63 69.5 72.3 3.52 2200000001000010000000000
131 0 0 28 1.70 78.2 98.6 2.84 1oooooooooooooooooooooooo
132 3 4 25 1.63 79.5 90.5 3.15 0I00000000000000000000010
133 1 6 22 1.63 60.5 77.7 2.39 0111000000000000000001000
134 1 7 42 1.50 44.5 48.6 2.27 5210000000100000000001001
135 1 8 21 1.60 60.5 73.2 3.27 0100000000000000000000100
136 2 4 20 1.52 60.5 75.0 3.64 00000I0000000001010000000
137 1 0 24 1.63 56.4 70.0 3.07 000000000000(8100000000000
138 1 5 33 1.65 40.0 55.0 3.47 0110000000010010000101000
139 1 6 33 1.65 63.6 82.7 3.86 0100010000010000000000000
140 3 3 29 1.63 58,2 65.5 2.84 1310000000000000000000000
141 2 8 27 1.65 57.3 69.5 3.64 0100010010110100000001000
142 3 3 20 1.52 44.5 51.4 1.93 1300000000100101010011111
143 2 6 18 1.63 49.1 56.8 3.24 000000000001OOOOOOOO10100
144 2 7 26 1.57 52.7 65.9 3.49 010001001010000I000000000
145 1 7 21 1.55 52.3 65.0 3.47 0100000010000000000001000
146 2 4 21 1.52 50.9 68.2 3.64 0100000010000101000001000
147 1 5 21 1.68 66.8 87.3 2.90 0100000110011101000000000
148 1 0 26 1.65 50.9 70.0 3.52 0200000000000000000000000
149 1 8 21 1.55 53.2 56.4 2.44 0100000010000000000000001
150 2 7 22 1.47 44.5 58.2 2.87 0100010000000101000010000
151 3 7 39 1.50 50.9 72.7 3.04 4200000IOOOOOOOOOOOOO1001
152 2 6 27 1.63 60.5 71.4 3.21 0100000LOOOOOOOOOO10I t000
153 1 0 24 1.57 60.0 72.7 3.35 0100000000000000000000000
154 1 4 24 1.63 64.5 75.0 2.87 0110000000000000000000100
155 1 7 19 1.55 52.7 57.7 2.61 010001OOOOOOOOQOOOOOOOO1o
156 1 0 22 1.60 67.3 73.6 2.05 0000000000000000000000000
157 3 8 27 1.55 60.5 73.2 3.98 0I0010000001OOOOOOOOOOOOO
158 3 2 35 1.73 72.3 89.1 6.28 3330010000001000000000100
159 0 0 24 1.65 83.6 86.8 3.64 0000000000000000000000000
160 1 0 35 1.55 52.3 59.1 3.64 010000001001OOOOOOOOOOOOO
161 1 0 21 1.52 59.1 67.3 2.95 01ooooooooooooooooooooooo
162 3 7 21 1.68 54.1 74.5 3.72 1200000010111101000001001
163 3 9 38 1.52 43.6 60.9 3.24 1410000000000000000000000
164 1 0 32 1.73 54.5 62.7 3,41 1300010010100000000001000
165 3 8 15 1.55 48.2 56.4 2.70 01100001001oooooooooooooo
166 2 6 23 1.57 60.5 100.0 3.58 0100010000100000000000000
167 2 7 19 1.50 49.5 63.6 3.35 OOOOO101OOOOOOOOOOOOO1101
168 2 8 25 1.60 59.5 72.3 2.73 1300000100101000000000000
169 2 5 22 1.50 53.6 63.2 3.35 0100011000000000000000100
170 0 0 19 1.55 57.3 65.0 3.07 1200000000000000000000000
171 1 0 23 1.68 57.3 61.4 3.58 0100000000000000000000000
172 3 0 36 1.63 54.1 60.0 2.84 0101000000I00000000001010
173 0 0 21 1.55 50.9 63.6 3.01 01ooooooooooooooooooooooo
174 0 0 30 1.52 38.2 48.2 2.56 01ooooooooooooooooooooooo
175 0 0 42 1.70 55.0 65.5 3.27 4200000000000000000000000
176 1 0 34 1.63 50.9 60.5 2.93 0100001000000000000000100
177 3 3 26 1.63 66.8 84.1 3.10 1300010000010000000001100
178 1 7 18 1.50 54.1 60.5 3.52 0000001000000000000000100
179 3 6 39 1.52 82.7 84.1 3,35 1210000000000000000000000
180 1 0 25 1.52 52.3 66.8 3.24 1200000000000010000010000
"""


def fix_ocr_comma(s):
    """OCR sometimes reads '.' as ','."""
    return s.replace(",", ".")


def normalize_packed(packed_raw):
    """
    Normalize the 25-char packed string from OCR.
    OCR artefacts: 'O'/'o' -> '0', 'I'/'L'/'l' -> '1',
    parenthetical noise like '(X)', '(8', 'Q', spaces, etc.
    Returns (cleaned_25_chars, list_of_ocr_issues).
    """
    issues = []
    original = packed_raw

    # Remove spaces
    s = packed_raw.replace(" ", "")

    # Remove parenthetical OCR noise: (X), (8, etc.
    s_before = s
    s = re.sub(r'\([^)]*\)', '0', s)
    s = re.sub(r'\(', '', s)
    s = re.sub(r'\)', '', s)
    if s != s_before:
        issues.append(f"parenthetical noise removed: '{s_before}' -> '{s}'")

    # Character substitutions
    result = []
    for ch in s:
        if ch in '0123456789':
            result.append(ch)
        elif ch in 'Oo':
            result.append('0')
        elif ch in 'ILl':
            result.append('1')
        elif ch == 't':
            # OCR artefact for '1'
            result.append('1')
        elif ch in 'QqCcRrKkMmNn':
            # Various OCR noise chars in packed strings -> '0'
            result.append('0')
        else:
            issues.append(f"unknown char '{ch}' replaced with '0'")
            result.append('0')

    cleaned = ''.join(result)

    if len(cleaned) != 25:
        issues.append(f"length {len(cleaned)} != 25 (from '{original}')")

    return cleaned, issues


def parse_ocr_lines():
    """Parse OCR text into list of dicts keyed by patient ID."""
    records = {}
    ocr_issues = {}

    for line in RAW_OCR.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Try to match the data pattern
        # ID(3) severity(1) month(1-2) age(2) height(F5.2) preWeight(F5.1) pregWeight(F5.1) babyWeight(F5.2) packed(25)
        # Use regex to be flexible with OCR artefacts
        m = re.match(
            r'(\d{3})\s+'          # patient ID
            r'(\d)\s+'             # severity
            r'(\d)\s+'             # month
            r'(\d+)\s+'            # age
            r'([0-9.,]+)\s+'       # height
            r'([0-9.,]+)\s+'       # preWeight
            r'([0-9.,]+)\s+'       # pregWeight
            r'([0-9.,]+)\s+'       # babyWeight
            r'(.+)$',              # packed string (rest of line)
            line
        )
        if not m:
            print(f"WARNING: Could not parse line: {line}", file=sys.stderr)
            continue

        pid = int(m.group(1))
        severity = int(m.group(2))
        month = int(m.group(3))
        age = int(m.group(4))
        height = float(fix_ocr_comma(m.group(5)))
        pre_weight = float(fix_ocr_comma(m.group(6)))
        preg_weight = float(fix_ocr_comma(m.group(7)))
        baby_weight = float(fix_ocr_comma(m.group(8)))

        packed_raw = m.group(9).strip()
        packed, issues = normalize_packed(packed_raw)

        if len(packed) < 25:
            packed = packed.ljust(25, '0')
            issues.append("padded to 25 chars")
        elif len(packed) > 25:
            issues.append(f"truncated from {len(packed)} to 25 chars")
            packed = packed[:25]

        parity = int(packed[0])
        prev_backache = int(packed[1])
        binary_cols = [int(packed[i]) for i in range(2, 25)]  # 23 binary: relief(8) + aggrav(15)

        record = {
            'id': pid,
            'col_2': severity,    # painSeverity
            'col_3': month,       # painOnsetMonth
            'col_4': age,
            'col_5': height,
            'col_6': pre_weight,
            'col_7': preg_weight,
            'col_8': baby_weight,
            'col_9': parity,
            'col_10': prev_backache,
        }
        # col_11 through col_33 = binary_cols[0..22]
        for i in range(23):
            col_name = f'col_{11 + i}'
            record[col_name] = binary_cols[i]

        records[pid] = record
        if issues:
            ocr_issues[pid] = issues

    return records, ocr_issues


def load_pmlb(path):
    """Load PMLB CSV into dict keyed by patient ID."""
    records = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(float(row['id']))
            record = {
                'id': pid,
                'col_2': int(row['col_2']),
                'col_3': int(row['col_3']),
                'col_4': int(float(row['col_4'])),
                'col_5': float(row['col_5']),
                'col_6': float(row['col_6']),
                'col_7': float(row['col_7']),
                'col_8': float(row['col_8']),
                'col_9': int(row['col_9']),
                'col_10': int(row['col_10']),
            }
            for i in range(11, 33):
                record[f'col_{i}'] = int(row[f'col_{i}'])
            # target = col_33 in Chatfield
            record['col_33'] = int(row['target'])
            records[pid] = record
    return records


def compare(ocr_records, pmlb_records, ocr_issues):
    """Compare records and report discrepancies."""
    all_cols = ['col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_8',
                'col_9', 'col_10'] + [f'col_{i}' for i in range(11, 34)]

    col_names = {
        'col_2': 'painSeverity', 'col_3': 'painOnsetMonth', 'col_4': 'age',
        'col_5': 'height', 'col_6': 'preWeight', 'col_7': 'pregWeight',
        'col_8': 'babyWeight', 'col_9': 'parity', 'col_10': 'prevBackache',
        'col_11': 'reliefTablets', 'col_12': 'reliefHotWaterBottle',
        'col_13': 'reliefHotBath', 'col_14': 'reliefCushion',
        'col_15': 'reliefStanding', 'col_16': 'reliefSitting',
        'col_17': 'reliefLying', 'col_18': 'reliefWalking',
        'col_19': 'aggravFatigue', 'col_20': 'aggravBending',
        'col_21': 'aggravLifting', 'col_22': 'aggravMakingBeds',
        'col_23': 'aggravWashingUp', 'col_24': 'aggravIroning',
        'col_25': 'aggravBowelAction', 'col_26': 'aggravIntercourse',
        'col_27': 'aggravCoughing', 'col_28': 'aggravSneezing',
        'col_29': 'aggravTurningInBed', 'col_30': 'aggravStanding',
        'col_31': 'aggravSitting', 'col_32': 'aggravLying',
        'col_33': 'aggravWalking(TARGET)',
    }

    float_cols = {'col_5', 'col_6', 'col_7', 'col_8'}

    total_discrepancies = 0
    ocr_uncertain = 0
    definite_discrepancies = []
    ocr_artefact_discrepancies = []

    missing_in_ocr = set(pmlb_records.keys()) - set(ocr_records.keys())
    missing_in_pmlb = set(ocr_records.keys()) - set(pmlb_records.keys())

    if missing_in_ocr:
        print(f"\nPatients in PMLB but missing from OCR: {sorted(missing_in_ocr)}")
    if missing_in_pmlb:
        print(f"\nPatients in OCR but missing from PMLB: {sorted(missing_in_pmlb)}")

    for pid in sorted(set(ocr_records.keys()) & set(pmlb_records.keys())):
        ocr = ocr_records[pid]
        pmlb = pmlb_records[pid]
        has_ocr_issues = pid in ocr_issues

        for col in all_cols:
            ocr_val = ocr.get(col)
            pmlb_val = pmlb.get(col)

            if ocr_val is None or pmlb_val is None:
                continue

            if col in float_cols:
                match = abs(ocr_val - pmlb_val) < 0.015
            else:
                match = ocr_val == pmlb_val

            if not match:
                total_discrepancies += 1
                name = col_names.get(col, col)
                entry = {
                    'pid': pid,
                    'col': col,
                    'name': name,
                    'ocr': ocr_val,
                    'pmlb': pmlb_val,
                    'has_ocr_issues': has_ocr_issues,
                }
                if has_ocr_issues:
                    ocr_artefact_discrepancies.append(entry)
                    ocr_uncertain += 1
                else:
                    definite_discrepancies.append(entry)

    return {
        'total': total_discrepancies,
        'definite': definite_discrepancies,
        'ocr_artefact': ocr_artefact_discrepancies,
        'ocr_uncertain_count': ocr_uncertain,
    }


def main():
    print("=" * 80)
    print("BACKACHE DATASET: TEXTBOOK vs PMLB COMPARISON")
    print("=" * 80)

    ocr_records, ocr_issues = parse_ocr_lines()
    print(f"\nParsed {len(ocr_records)} records from OCR text")

    pmlb_records = load_pmlb('/Users/mike/dev/explaneat/scripts/backache_pmlb.csv')
    print(f"Loaded {len(pmlb_records)} records from PMLB")

    print(f"\nRecords with OCR artefacts (unreliable comparison): {len(ocr_issues)}")
    for pid, issues in sorted(ocr_issues.items()):
        print(f"  Patient {pid:3d}: {'; '.join(issues)}")

    results = compare(ocr_records, pmlb_records, ocr_issues)

    print(f"\n{'=' * 80}")
    print(f"COMPARISON RESULTS")
    print(f"{'=' * 80}")
    print(f"Total discrepancies: {results['total']}")
    print(f"  - In OCR-clean rows (definite): {len(results['definite'])}")
    print(f"  - In OCR-artefact rows (uncertain): {results['ocr_uncertain_count']}")

    if results['definite']:
        print(f"\n--- DEFINITE DISCREPANCIES (clean OCR rows) ---")
        for d in results['definite']:
            print(f"  Patient {d['pid']:3d}, {d['col']} ({d['name']}): "
                  f"textbook={d['ocr']}, PMLB={d['pmlb']}")

    if results['ocr_artefact']:
        print(f"\n--- OCR-ARTEFACT DISCREPANCIES (unreliable, may be OCR errors) ---")
        for d in results['ocr_artefact']:
            print(f"  Patient {d['pid']:3d}, {d['col']} ({d['name']}): "
                  f"textbook={d['ocr']}, PMLB={d['pmlb']}")

    # Summary statistics
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    n_clean = len(ocr_records) - len(ocr_issues)
    n_total_cols = 32  # 32 data columns per row
    print(f"Clean rows (no OCR artefacts): {n_clean} of {len(ocr_records)}")
    print(f"Total cells compared in clean rows: {n_clean * n_total_cols}")
    print(f"Definite discrepancies: {len(results['definite'])}")
    if n_clean > 0:
        pct = len(results['definite']) / (n_clean * n_total_cols) * 100
        print(f"Discrepancy rate (clean rows): {pct:.2f}%")

    # Check the specific concern: is col_33 really just an aggravating factor?
    print(f"\n{'=' * 80}")
    print(f"TARGET COLUMN ANALYSIS")
    print(f"{'=' * 80}")
    target_ones = sum(1 for r in pmlb_records.values() if r['col_33'] == 1)
    target_zeros = sum(1 for r in pmlb_records.values() if r['col_33'] == 0)
    print(f"PMLB target=1 (walking aggravates): {target_ones}")
    print(f"PMLB target=0: {target_zeros}")

    # Check: of the target=1 patients, how many have painSeverity > 0?
    target1_with_pain = sum(1 for r in pmlb_records.values()
                           if r['col_33'] == 1 and r['col_2'] > 0)
    target1_no_pain = sum(1 for r in pmlb_records.values()
                         if r['col_33'] == 1 and r['col_2'] == 0)
    print(f"  Of target=1: {target1_with_pain} have pain (severity>0), "
          f"{target1_no_pain} have no pain (severity=0)")

    # Severity distribution
    print(f"\nPain severity distribution:")
    for sev in range(4):
        n = sum(1 for r in pmlb_records.values() if r['col_2'] == sev)
        print(f"  Severity {sev}: {n} patients")

    severe = sum(1 for r in pmlb_records.values() if r['col_2'] == 3)
    print(f"\nSevere backache (severity=3): {severe} patients")
    print(f"Target=1 (walking aggravates): {target_ones} patients")
    print(f"These are {'DIFFERENT' if severe != target_ones else 'THE SAME'} counts.")


if __name__ == '__main__':
    main()
