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
//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \\file {filename}
//---------------------------------------------------------------------------//
'''

HEADER_FILE = TOP + '''\
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

#include "{name}.i.h"

#endif // {header_guard}
'''

INLINE_FILE = TOP + '''\
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

CODE_FILE = TOP + '''\
namespace celeritas {{
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
}}  // namespace celeritas
'''

TEMPLATES = {
    'h': HEADER_FILE,
    'i.h': INLINE_FILE,
    'cc': CODE_FILE
}

def generate(root, filename):
    if os.path.exists(filename):
        print("Skipping existing file " + filename)
        return
    relpath = os.path.relpath(filename, start=root)
    (basename, _, ext) = filename.partition('.')
    try:
        template = TEMPLATES[ext]
    except KeyError:
        print("Invalid extension ." + ext)
        sys.exit(1)

    variables = {
        'name': re.sub(r'\..*', '', os.path.basename(filename)),
        'header_guard': re.sub(r'\W', '_', relpath),
        'filename': filename,
        }
    with open(filename, 'w') as f:
        f.write(template.format(**variables))

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
