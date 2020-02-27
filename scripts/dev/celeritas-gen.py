#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
# File: scripts/dev/celeritas-gen.py
###############################################################################
"""Generate class file stubs for Celeritas.
"""
from __future__ import (division, absolute_import, print_function)
import os.path
import re
import subprocess
import sys
###############################################################################

TOP = '''\
//{modeline:-^75s}//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \\file {filename}
//---------------------------------------------------------------------------//
'''

HEADER_FILE = '''\
#ifndef {header_guard}
#define {header_guard}

namespace celeritas {{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
  {name} ...;
   \endcode
 */
class {name} {{
 public:
  //@{{
  //! Type aliases
  <++>
  //@}}

 public:
  // Construct with defaults
  inline {name}();
}};

//---------------------------------------------------------------------------//
}}  // namespace celeritas

#include "{name}.i.{hext}"

#endif // {header_guard}
'''

INLINE_FILE = '''\

namespace celeritas {{
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
{name}::{name}() {{
}}

//---------------------------------------------------------------------------//
}}  // namespace celeritas
'''

CODE_FILE = '''\
#include "{name}.{hext}"

namespace celeritas {{
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
}}  // namespace celeritas
'''

TEMPLATES = {
    'h': HEADER_FILE,
    'i.h': INLINE_FILE,
    'cc': CODE_FILE,
    'cu': CODE_FILE,
    'cuh': HEADER_FILE,
    'k.cuh': INLINE_FILE,
    'i.cuh': INLINE_FILE,
    't.cuh': INLINE_FILE,
}

LANG = {
    'h': "C++",
    'cc': "C++",
    'cu': "CUDA",
    'cuh': "CUDA",
}

HEXT = {
    'C++': "h",
    'CUDA': "cuh",
}

def generate(root, filename):
    if os.path.exists(filename):
        print("Skipping existing file " + filename)
        return
    relpath = os.path.relpath(filename, start=root)
    (basename, _, longext) = filename.partition('.')
    try:
        template = TEMPLATES[longext]
    except KeyError:
        print("Invalid extension ." + longext)
        sys.exit(1)

    ext = longext.split('.')[-1]
    lang = LANG[ext]

    variables = {
        'longext': longext,
        'ext': ext,
        'hext': HEXT[lang],
        'modeline': "-*-{}-*-".format(lang),
        'name': re.sub(r'\..*', '', os.path.basename(filename)),
        'header_guard': re.sub(r'\W', '_', relpath),
        'filename': filename,
        }
    with open(filename, 'w') as f:
        f.write((TOP + template).format(**variables))

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename', nargs='+',
                        help='file names to generate')
    parser.add_argument('--basedir',
                        help='root source directory for file naming')
    args = parser.parse_args()
    basedir = args.basedir or os.path.join(
            subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
                .decode().strip(),
            'src')
    for fn in args.filename:
        generate(basedir, fn)

#-----------------------------------------------------------------------------#
if __name__ == '__main__':
    main()

###############################################################################
# end of scripts/dev/celeritas-gen.py
###############################################################################
