# .claude

Global configuration and instructions for Claude Code.

## Setup

Go to your home directory:
```sh
cd ~
```

If `~/.claude` does not exist, run the following command to clone this repo and create it.

```sh
git clone git@github.com:eringrant/.claude
```

If you receive the error `fatal: destination path '.claude' already exists and is not an empty directory.`
(i.e., you already have a `.claude` directory in your home directory),
then you can run the following commands to consolidate the existing directory with this repo:

```sh
git init
git remote add origin git@github.com:eringrant/.claude
git fetch
git pull origin main
```

You will be prompted to merge files if necessary.
