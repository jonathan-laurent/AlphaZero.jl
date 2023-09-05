#!/bin/bash
#
# Automated install, unzip, and link Julia 1.9.3
#
# The present script downloads Julia, decompress
# and unzip it into $HOME/sw. Then it removes the
# downloaded archive and adds a symbolic link to
# julia binary executable.
#
#

# delete the folders "bin" (binaries) and "sw" (software) in $HOME if they exist
rm -rf $HOME/bin
rm -rf $HOME/sw
rm -rf $HOME/.julia

# Let's create two folders: "bin" (binaries) and "sw" (software) in $HOME
mkdir -pv $HOME/bin
mkdir -pv $HOME/sw
mkdir -pv $HOME/.julia/config


# Now let's download and unpack Julia
wget --no-check-certificate -O julia.tar.gz https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.3-linux-x86_64.tar.gz
gzip -d julia.tar.gz
tar -xf julia.tar

# Let's clean up, removing the downloaded archive, as it is no longer needed
rm -f julia.tar

mv julia-1.9.3 $HOME/sw/

# Let's finally create a symbolic link, so that julia's binary can be easy found
rm -f $HOME/bin/julia
ln -s $HOME/sw/julia-1.9.3/bin/julia $HOME/bin/julia

# Add the path to .bashrc
echo "export PATH=$HOME/bin:$PATH" >> ~/.bashrc

# Add the path to the current session
export PATH=$HOME/bin:$PATH

echo 'ENV["PYTHON"] = "'$HOME'/sw/miniconda3/bin/python3"' >> ~/.julia/config/startup.jl
echo 'println("Version: 1.9.3")' >> ~/.julia/config/startup.jl
