"""The sales-order graph from notebook 1 and the path and motif encoders, for later notebooks to import."""

import os

os.environ.setdefault("LANCEDB_LOG", "error")  # hide LanceDB's first-write WARN; import this before lancedb

import torch
import torchhd
from lance_csr import GraphConfig

# Every node has an integer id, which lance-csr needs to build its CSR adjacency, and a readable name.
# Ids are numbered by label: customers 1xx, products 2xx, categories 3xx, depots 4xx, regions 5xx,
# countries 6xx, issues 7xx. An order's id is its order number, so SO-1042 is 1042.
NODES = {
    "Customer": [
        {"id": 101, "name": "Liffey Interiors"},
        {"id": 102, "name": "Lee Valley Homes"},
        {"id": 103, "name": "Atlantic Living"},
        {"id": 104, "name": "Maison du Nord"},
        {"id": 105, "name": "Northern Nest"},
        {"id": 106, "name": "Pennine House"},
    ],
    "Order": [
        {"id": 901, "name": "SO-0901", "status": "resolved"},
        {"id": 903, "name": "SO-0903", "status": "resolved"},
        {"id": 904, "name": "SO-0904", "status": "resolved"},
        {"id": 906, "name": "SO-0906", "status": "resolved"},
        {"id": 907, "name": "SO-0907", "status": "resolved"},
        {"id": 908, "name": "SO-0908", "status": "resolved"},
        {"id": 910, "name": "SO-0910", "status": "resolved"},
        {"id": 912, "name": "SO-0912", "status": "resolved"},
        {"id": 914, "name": "SO-0914", "status": "resolved"},
        {"id": 915, "name": "SO-0915", "status": "resolved"},
        {"id": 917, "name": "SO-0917", "status": "resolved"},
        {"id": 920, "name": "SO-0920", "status": "resolved"},
        {"id": 1042, "name": "SO-1042", "status": "open"},
    ],
    "Product": [
        {"id": 201, "name": "cordless-table-lamp"},
        {"id": 202, "name": "brass-floor-lamp"},
        {"id": 203, "name": "velvet-sofa"},
        {"id": 204, "name": "corner-sofa"},
        {"id": 205, "name": "wool-rug"},
        {"id": 206, "name": "jute-rug"},
        {"id": 207, "name": "oak-desk"},
        {"id": 208, "name": "standing-desk"},
    ],
    "Category": [
        {"id": 301, "name": "lamps"},
        {"id": 302, "name": "sofas"},
        {"id": 303, "name": "rugs"},
        {"id": 304, "name": "desks"},
    ],
    "Depot": [
        {"id": 401, "name": "east-london"},
        {"id": 402, "name": "west-london"},
        {"id": 403, "name": "midlands"},
    ],
    "Region": [
        {"id": 501, "name": "greater-london"},
        {"id": 502, "name": "west-midlands"},
        {"id": 503, "name": "greater-manchester"},
        {"id": 504, "name": "west-yorkshire"},
        {"id": 505, "name": "merseyside"},
        {"id": 506, "name": "dublin"},
        {"id": 507, "name": "cork"},
        {"id": 508, "name": "galway"},
        {"id": 509, "name": "limerick"},
        {"id": 510, "name": "ile-de-france"},
        {"id": 511, "name": "hauts-de-france"},
    ],
    "Country": [
        {"id": 601, "name": "uk"},
        {"id": 602, "name": "ireland"},
        {"id": 603, "name": "france"},
    ],
    "Issue": [
        {"id": 701, "name": "customs_delay"},
        {"id": 702, "name": "dispatch_hold"},
        {"id": 703, "name": "damaged_in_transit"},
    ],
}

EDGES = {
    "PLACED": [
        (101, 901),  # liffey-interiors → SO-0901
        (102, 904),  # lee-valley-homes → SO-0904
        (103, 908),  # atlantic-living → SO-0908
        (104, 912),  # maison-du-nord → SO-0912
        (104, 915),  # maison-du-nord → SO-0915
        (102, 920),  # lee-valley-homes → SO-0920
        (105, 906),  # northern-nest → SO-0906
        (105, 910),  # northern-nest → SO-0910
        (102, 917),  # lee-valley-homes → SO-0917
        (106, 903),  # pennine-house → SO-0903
        (105, 907),  # northern-nest → SO-0907
        (106, 914),  # pennine-house → SO-0914
        (101, 1042),  # liffey-interiors → SO-1042
    ],
    "CONTAINS": [
        (901, 203),  # SO-0901 → velvet-sofa
        (904, 205),  # SO-0904 → wool-rug
        (908, 207),  # SO-0908 → oak-desk
        (912, 202),  # SO-0912 → brass-floor-lamp
        (915, 206),  # SO-0915 → jute-rug
        (920, 208),  # SO-0920 → standing-desk
        (906, 204),  # SO-0906 → corner-sofa
        (910, 203),  # SO-0910 → velvet-sofa
        (917, 201),  # SO-0917 → cordless-table-lamp
        (903, 202),  # SO-0903 → brass-floor-lamp
        (907, 202),  # SO-0907 → brass-floor-lamp
        (914, 207),  # SO-0914 → oak-desk
        (1042, 201),  # SO-1042 → cordless-table-lamp
    ],
    "IN_CATEGORY": [
        (201, 301),  # cordless-table-lamp → lamps
        (202, 301),  # brass-floor-lamp → lamps
        (203, 302),  # velvet-sofa → sofas
        (204, 302),  # corner-sofa → sofas
        (205, 303),  # wool-rug → rugs
        (206, 303),  # jute-rug → rugs
        (207, 304),  # oak-desk → desks
        (208, 304),  # standing-desk → desks
    ],
    "SHIPS_FROM": [
        (901, 401),  # SO-0901 → east-london
        (904, 403),  # SO-0904 → midlands
        (908, 402),  # SO-0908 → west-london
        (912, 403),  # SO-0912 → midlands
        (915, 402),  # SO-0915 → west-london
        (920, 401),  # SO-0920 → east-london
        (906, 403),  # SO-0906 → midlands
        (910, 403),  # SO-0910 → midlands
        (917, 401),  # SO-0917 → east-london
        (903, 402),  # SO-0903 → west-london
        (907, 403),  # SO-0907 → midlands
        (914, 402),  # SO-0914 → west-london
        (1042, 401),  # SO-1042 → east-london
    ],
    "SHIPS_TO": [
        (901, 506),  # SO-0901 → dublin
        (904, 507),  # SO-0904 → cork
        (908, 508),  # SO-0908 → galway
        (912, 510),  # SO-0912 → ile-de-france
        (915, 511),  # SO-0915 → hauts-de-france
        (920, 507),  # SO-0920 → cork
        (906, 502),  # SO-0906 → west-midlands
        (910, 503),  # SO-0910 → greater-manchester
        (917, 507),  # SO-0917 → cork
        (903, 504),  # SO-0903 → west-yorkshire
        (907, 502),  # SO-0907 → west-midlands
        (914, 505),  # SO-0914 → merseyside
        (1042, 506),  # SO-1042 → dublin
    ],
    "IN_COUNTRY": [
        (501, 601),  # greater-london → uk
        (502, 601),  # west-midlands → uk
        (503, 601),  # greater-manchester → uk
        (504, 601),  # west-yorkshire → uk
        (505, 601),  # merseyside → uk
        (506, 602),  # dublin → ireland
        (507, 602),  # cork → ireland
        (508, 602),  # galway → ireland
        (509, 602),  # limerick → ireland
        (510, 603),  # ile-de-france → france
        (511, 603),  # hauts-de-france → france
    ],
    # (order, issue, note): the note is a property on the edge
    "HAD_ISSUE": [
        (901, 701, "Commercial invoice missing; Irish customs held the parcel until the broker resent it."),
        (904, 701, "Wrong commodity code on the customs declaration; corrected and re-filed."),
        (908, 701, "Import VAT unpaid; released after the customer paid it."),
        (912, 701, "Held at Calais: EORI number missing from the paperwork."),
        (915, 701, "French customs asked for proof of origin; supplier certificate sent."),
        (920, 701, "Customs declaration lodged late by the broker."),
        (906, 702, "Sofa out of stock at the Midlands depot; held until the next supplier delivery."),
        (910, 702, "Held at the Midlands depot waiting for a two-person delivery slot."),
        (917, 702, "Cordless lamps (lithium batteries) held at East London depot for dangerous-goods paperwork; released after calling the depot."),
        (903, 703, "Lamp shade crushed in transit; replacement sent."),
        (907, 703, "Lamp base cracked; carrier claim filed."),
        (914, 703, "Desk top scratched on arrival; carrier claim filed."),
    ],
}

# Notebook 4: orders resolved this week, labelled but not yet in the graph or the memory.
HELD_OUT = [
    {"order": "SO-1051", "category": "lamps", "depot": "east-london", "region": "limerick", "country": "ireland", "issue": "dispatch_hold"},
    {"order": "SO-1053", "category": "lamps", "depot": "east-london", "region": "greater-london", "country": "uk", "issue": "dispatch_hold"},
    {"order": "SO-1055", "category": "desks", "depot": "east-london", "region": "dublin", "country": "ireland", "issue": "customs_delay"},
    {"order": "SO-1057", "category": "lamps", "depot": "west-london", "region": "west-midlands", "country": "uk", "issue": "damaged_in_transit"},
    {"order": "SO-1058", "category": "rugs", "depot": "midlands", "region": "ile-de-france", "country": "france", "issue": "customs_delay"},
    {"order": "SO-1060", "category": "sofas", "depot": "midlands", "region": "greater-manchester", "country": "uk", "issue": "dispatch_hold"},
]

REL_ENDPOINTS = {
    "PLACED": ("Customer", "Order"),
    "CONTAINS": ("Order", "Product"),
    "IN_CATEGORY": ("Product", "Category"),
    "SHIPS_FROM": ("Order", "Depot"),
    "SHIPS_TO": ("Order", "Region"),
    "IN_COUNTRY": ("Region", "Country"),
    "HAD_ISSUE": ("Order", "Issue"),
}


def edge_rows(rel):
    if rel == "HAD_ISSUE":
        return [{"src": s, "dst": d, "note": n} for s, d, n in EDGES[rel]]
    return [{"src": s, "dst": d} for s, d in EDGES[rel]]


def write_graph(db):
    for label, rows in NODES.items():
        db.create_table(label, rows, mode="overwrite")
    for rel in EDGES:
        db.create_table(rel, edge_rows(rel), mode="overwrite")


def graph_config():
    builder = GraphConfig.builder()
    for label in NODES:
        builder = builder.node_table(label, "id")
    for rel, (src, dst) in REL_ENDPOINTS.items():
        builder = builder.rel_table(rel, src, dst, "src", "dst")
    return builder.build()


def make_codebook(D=10_000, seed=0):
    """We give every symbol in the graph its own random hypervector, and return them as a dict.

    Why we need a codebook: HDC only computes with hypervectors, so names like "ireland" must be
    turned into hypervectors before we can bind or bundle them. The codebook acts as a lookup table,
    so that we can turn any symbol into a hypervector and decode them back into their original forms.

    The symbols are every node name (such as "ireland" or "SO-1042") and every relationship type
    (such as "SHIPS_TO"). Each gets a random ±1 vector of length D, and random vectors this long
    are nearly orthogonal, so no two symbols look alike. The codebook is the dictionary that
    every other hypervector is built from with ⊗, ⊕ and ρ, and that results are decoded against.
    The fixed seed gives the same codebook on every run.
    """
    torch.manual_seed(seed)
    names = [n["name"] for rows in NODES.values() for n in rows] + list(EDGES)
    return dict(zip(names, torchhd.random(len(names), D, vsa="MAP")))


def path(codebook, *steps):
    """r1 ⊗ ρ(r2) ⊗ … ⊗ ρ^(k-1)(rk) ⊗ ρ^k(end node): the last step is the end node."""
    hv = codebook[steps[0]]
    for k, step in enumerate(steps[1:], start=1):
        hv = torchhd.bind(hv, torchhd.permute(codebook[step], shifts=k))
    return hv


def motif(codebook, category=None, depot=None, region=None, country=None):
    """Bundle the order's known paths. Leave a field as None if it's unknown."""
    paths = []
    if category:
        paths.append(path(codebook, "CONTAINS", "IN_CATEGORY", category))
    if depot:
        paths.append(path(codebook, "SHIPS_FROM", depot))
    if region:
        paths.append(path(codebook, "SHIPS_TO", region))
    if country:
        paths.append(path(codebook, "SHIPS_TO", "IN_COUNTRY", country))
    return torchhd.multiset(torch.stack(paths))
