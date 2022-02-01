/*
	################################################################################
	Sript built to restore Ax2012.bak file into SQLMI
	Author: Nick Tinsley
	Date: 11/18/2020
	Ver: 1.0
	Change Log: 2-27-2021 updated to point and connect to Prod
	################################################################################
*/

/*
	create credential to connect to blob storage
	only needed to be done once, secret is removed from clear text

Create credential [https://cmdwsa1prod.blob.core.windows.net/cm-dw-container1-prod] --[https://cmdwdatalakenorthcentral.blob.core.windows.net/cmdwdatalakencentral]
	with IDENTITY = 'SHARED ACCESS SIGNATURE',
	SECRET = '[SAS' 
			--to create secret you must create an access policy first
			--get access key from policy,, do not include the ?
GO
*/

--###################################################################################
/*
	following command lists the contents of the storage account, this is to troubleshoot and make sure 
	the connection was created properly
RESTORE FILELISTONLY 
	FROM URL = N''

*/


--###################################################################################
/*
	following commands to restore backup from blob storage.  This will need to be executed again when AX2012 is rested.
	takes about 1 hour to run so be patient

DROP DATABASE AX2012
Restore DATABASE NAMEOFDB
	FROM URL = N'';


--ax2012 model 2.5 mins --not needed for prod
Restore DATABASE AX2012_Model
	FROM URL = N'https://cmdwsa1.blob.core.windows.net/cm-dw-container-1/AX2012R2PROD_model.bak';

--axDW 2012 -- 15 mins --needed for fusion
drop database axdw
RESTORE DATABASE AXDW
	FROM URL = N'https://cmdwsa1prod.blob.core.windows.net/cm-dw-container1-prod/DW_Analytical.bak';

Restore DATABASE Wolfe_Pak
	FROM URL = N'https://cmdwsa1prod.blob.core.windows.net/cm-dw-container1-prod/WolfePak.bak';

*/
