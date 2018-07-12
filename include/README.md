# include

In this folder we maintain symbolic links to all headers in src, to allow flat include statements.

If you add or remove source files in src, you should run the following from project root:
  `make update-sources`

This will update the lib and include folders to a changed src.
