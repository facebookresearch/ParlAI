Task: Holl_E
===============
Description: Sequence of utterances and responses with background knowledge about movies. From the Holl-E dataset. More information found at https://github.com/nikitacs16/Holl-E

**Command Line Arguments:**

`--knowledge-types`, `-kt` is a string of only `full`, `none` or should be a comma separated list of the following categories: plot, review, comments, fact_table, for example:
 `-kt full`, `-kt plot,review`, `-kt comments`. Defaults to `full`

*Category Descriptions:*

none: No knowledge used

full: All of the below

 plot: Movie plot description

 review: Movie review

 comments: Movie comments
 
 fact_table: Data containing box office earnings, similar movies, taglines, awards.

 More information can be found at https://github.com/nikitacs16/Holl-E/blob/main/data%20documentation/raw_data.md
 

**Tags: #All, #ChitChat**

