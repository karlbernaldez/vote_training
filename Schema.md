\------------------------------------------------
**Dataset**



\- id

\- source/model 	       FK / string              
- timestep             int / string

\- timestamp            date / string
- path                 string

\- domain\_id            // FK -> Domain ID 

\- start\_date	       date / string

\- end\_date	       date / string

\- format               grib, netcdf, json, tif

\- type                 Enum / string (gridded / spectra)

\- status 	       int / string

\- author\_id            // FK -> Author ID







\---------------------------------------------------
**Author**



\- id

\- First Name

\- Last Name





\--------------------------------------------------

**Domains** 



\- id

\- range (array) / min, max

\- elevation



\--------------------------------------------------

**Variables**



\- id

\- code

\- name

\- unit

\- value\_type

&#x09;

\-------------------------------------------------

Dataset Variables



&#x20;- id

&#x20;- dataset\_id - FK -> Dataset ID

&#x20;- variable\_ids (array) - FK -> Variable ID





\-------------------------------------------------
?????


**Variable Map**



&#x20;- id

&#x20;- model\_name / source id - string / FK Source ID

&#x20;- variable\_id - FK -> Variable ID

&#x20;- code 

