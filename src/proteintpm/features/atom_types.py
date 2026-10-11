import logging
import re

logger = logging.getLogger(__name__)

# Atom names already have been warned about
_WARNED_UNKNOWN: set[str] = set()


def normalize_atom_name(atom: str) -> str | None:
    """
    Return the atom name with digits removed, or None if invalid.
 
    Examples:
        "OE1"  -> "OE"
        "1HG2" -> "HG"
        "OXT"  -> "OXT"   (valid string; still unrecognized by map_atom -> dropped there)
        ""     -> None
        "123"  -> None
 
    Returning None signals the caller to drop the atom (same policy as HETATM
    and alternate-location records: strip and move on).
    """
    if not isinstance(atom, str):
        return None
 
    atom = atom.strip().upper()
    atom = re.sub(r"\d+", "", atom)  # remove all digits
 
    if not atom or not re.fullmatch(r"[A-Z]+", atom):
        return None
 
    return atom



def map_atom(atom, resid):
    """
    Map a target atom (given its atom name and residue) to a chemical atom-class
    code. Class codes are offset by +1 relative to the original scheme so that
    class 0 is reserved for zero-padding downstream (see stacking).
 
    Returns:
        int  : the atom-class code, if recognized.
        None : if the atom name is malformed, or is a valid but unrecognized
               atom (e.g. terminal OXT). Such atoms are dropped by the caller.
 
    Note:
        - HETATM records (waters, ions, etc.) are removed upstream, so no water
          oxygen class is defined here; anything unexpected returns None.

    """

    name = normalize_atom_name(atom)
    if name is None:
        return None
    
    if resid is None:
        # HETATM (incl. water) is stripped upstream; nothing valid reaches here.
        return None


    match name:
        case "CA" | "CB":
            return 0 + 1
        case "C":
            return 3 + 1
        case "CG":
            match resid:
                case (
                    "GLN"
                    | "GLU"
                    | "ILE"
                    | "LEU"
                    | "LYS"
                    | "MET"
                    | "PRO"
                    | "THR"
                    | "VAL"
                    | "ARG"
                ):
                    return 0 + 1
                case "PHE" | "TYR" | "TRP" | "HIS":
                    return 1 + 1
                case "ASN":
                    return 10 + 1
                case "ASP":
                    return 11 + 1
                case _:
                    pass
        case "CD":
            match resid:
                case "ARG" | "ILE" | "LEU" | "LYS" | "PRO":
                    return 0 + 1
                case "GLN":
                    return 10 + 1
                case "GLU":
                    return 11 + 1
                case "PHE" | "TYR" | "TRP" | "HIS":
                    return 1 + 1
        case "CE":
            match resid:
                case "HIS" | "PHE" | "TRP" | "TYR":
                    return 1 + 1
                case "LYS" | "MET":
                    return 0 + 1
        case "CZ":
            match resid:
                case "PHE" | "TRP" | "TYR":
                    return 1 + 1
                case "ARG":
                    return 15 + 1
        case "CH":
            return 1 + 1
        case "SG":
            return 14 + 1
        case "SD":
            return 13 + 1
        case "O":
            return 19 + 1
        case "OD":
            match resid:
                case "ASN":
                    return 7 + 1
                case "ASP":
                    return 5 + 1
        case "OG":
            return 4 + 1
        case "OE":
            match resid:
                case "GLN":
                    return 7 + 1
                case "GLU":
                    return 5 + 1
        case "OH":
            return 6 + 1
        case "N":
            return 2 + 1
        case "NE":
            match resid:
                case "ARG":
                    return 17 + 1
                case "HIS":
                    return 16 + 1
                case "TRP":
                    return 12 + 1
                case "GLN":
                    return 8 + 1
        case "NH":
            return 18 + 1
        case "ND":
            match resid:
                case "ASN":
                    return 8 + 1
                case "HIS":
                    return 16 + 1
        case "NZ":
            return 18 + 1

    if name not in _WARNED_UNKNOWN:
        _WARNED_UNKNOWN.add(name)
        logger.warning("Dropping unrecognized atom type: %s (resid=%s)", name, resid)
    return None


