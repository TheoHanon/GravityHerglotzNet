KPL/MK

   This meta-kernel lists a subset of kernels from the meta-kernel
   mro_2018_v04.tm provided in the data set MRO-M-SPICE-6-V1.0,
   covering the whole or a part of the customer requested time period
   from 2018-10-01T00:00:00.000 to 2018-10-31T00:00:00.000.

   The documentation describing these kernels can be found in the
   complete data set available at this URL

   ftp://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000

   To use this meta-kernel users may need to modify the value of the
   PATH_VALUES keyword to point to the actual location of the data
   set's ``data'' directory on their system. Replacing ``/'' with ``\''
   and converting line terminators to the format native to the user's
   system may also be required if this meta-kernel is to be used on a
   non-UNIX workstation.

   This meta-kernel was created by the NAIF node's SPICE PDS data set 
   subsetting service version 1.2 on Tue Apr 30 14:23:12 PDT 2019.

 
   \begindata
 
      PATH_VALUES     = (
                         'spice/data'
                        )
 
      PATH_SYMBOLS    = (
                         'KERNELS'
                        )
 
      KERNELS_TO_LOAD = (
                         '$KERNELS/naif0012.tls'
                         '$KERNELS/PCK00010.TPC'
                         
                        )
 
   \begintext
 

